import socket
import torch
import torch.distributed as dist
import logging
from utils import comm
import torch.multiprocessing as mp
#找到空闲的port，用于distributed data pallara的初始化
def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

#初始化ddp，特性local_rank必须放第一个
def _distributed_worker(main_func,local_rank, world_size, num_gpus_per_machine, machine_rank, dist_url, args):
    print(local_rank)
    #假如没有cuda报错
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    #算出作为主卡的卡的rank,机子序号（从0开始）*每个机子上有几张卡+它是本地第几张卡（单机的话就是0）
    global_rank=machine_rank*num_gpus_per_machine+local_rank
    #print(global_rank)
    try:
        #初始化
        dist.init_process_group(backend='NCCL',init_method=dist_url,world_size=world_size,rank=global_rank)
    #跑不通则报警
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    #synchronize is needed here to prevent a possible timeout after calling init_process_group
    #同步各卡之间的数据
    comm.synchronize()

    #每台机器上设置能用的卡数不能超过机器上的总卡数
    assert num_gpus_per_machine <= torch.cuda.device_count()
    #设置主卡
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)？？？
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg
    #然后开始训练
    #main_func(*args)


#DDPmode的初始化，默认单机多卡（不知道为啥，这里args必须设置一个默认值）
def launch(main_func,local_rank,num_gpus_per_machine, num_machines=1, machine_rank=0,dist_url=None,args=()):
    world_size=num_machines*num_gpus_per_machine
    #多卡模式
    if world_size>1:
        #确定卡间交流的port
        if dist_url=='auto':
            #自动找port只支持单机多卡
            assert num_machines==1, 'dist_url=auto not supported in multi-machine jobs.'
            port=_find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"    #f是另一种string相加的方式（把{}括号内表达式变成string）
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        #torch multi-process的方法，相当于用多进程执行_distributed_worker函数，并执行main函数（即训练函数），第二项是初始化_distributed_worker必要的，后面是参数
        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(main_func,local_rank,world_size, num_gpus_per_machine, machine_rank, dist_url,args),
            daemon=False,
        )
    else:
        main_func(*args)