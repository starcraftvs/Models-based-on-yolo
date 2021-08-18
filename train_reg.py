import torch
from utils.data.datasets import Reg_DataSet
from torch.utils.data import DataLoader
from utils.data.transform_factory import create_transform
from models.build_model import Model
from utils.data.classify_datasets import *
from utils.utils import *
from utils.data.dataloader import *
import numpy as np
from utils.data.transform_factory import *
from utils.utils import *
import yaml
import torch.nn as nn
from utils.defaults import *
from utils import torch_utils
import torch.distributed as dist

def train_reg(args):
    #是否混合精度训练
    mixed_precision = True
    try:  # Mixed precision training https://github.com/NVIDIA/apex
        from apex import amp
    except:
        print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed
    
    #训练超参,目前只用lr
    hyp = {'lr0': 0.01
        }  
    # 如果有hyp*.txt文件，就Overwrite hyp with hyp*.txt (optional)
    f = glob.glob('hyp*.txt')
    if f:
        print('Using %s' % f[0])
        for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
            hyp[k] = v
    print(hyp)
    
    
    #获取参数
    epochs=args.epochs
    output_dir=args.output_dir
    epochs = args.epochs  # 300
    batch_size = args.batch_size  # 16

    #设置保存模型及log的路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file=os.path.join(output_dir,'log.txt')
    #最后一次结果的路径
    last_path=os.path.join(output_dir,'last.pt')
    
    #设置device
    device = torch_utils.select_device(args.device, apex=mixed_precision, batch_size=args.batch_size)
    
    #建立模型
    model=Model(args.model_path).to(device)
    #loss function
    loss_fn=nn.SmoothL1Loss()
    #优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'])
    
    #是否load
    if args.resume:
        ckpt=torch.load(last_path,map_location=device)
        ckpt['model'] = \
            {k: v for k, v in ckpt['model'].state_dict().items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch=ckpt['epoch'] + 1
    else:
        start_epoch=0
    # Exponential moving average
    ema = torch_utils.ModelEMA(model)
    

    
    #如果可以，初始化单机多卡的环境
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # distributed backend
                                init_method='tcp://127.0.0.1:9999',  # init method
                                world_size=1,  # number of nodes
                                rank=0)  # node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
    #设置随机种子数，使得随机化的数都一样，每次训练可复现，而且不设置为0的话会自动搜索最合适的计算方法，加速计算
    init_seeds(1)
    
    #获取数据，并transform
    transforms=create_transform(args.input_size)
    train_dataset=Reg_DataSet(root_dir=args.train_dir,transforms=transforms,data_dir=args.data_dir,label_dir=args.label_dir)

    #load数据
    train_loader=DataLoader(train_dataset,batch_size=batch_size)

    #数据量
    last_idx=len(train_dataset)-1


    #跑多少个epoch
    i=0 #记录跑了多少iteration
    for epoch in range(start_epoch,epochs):
        running_loss=0
        #每个epoch里跑了多少个ieration
        it=0
        for batch_idx, (input, target) in enumerate(train_loader):
            last_batch = batch_idx == last_idx
            predict=model(input.to(device))

            loss=loss_fn(predict,target.to(device))
            loss.backward()
            optimizer.step()
            i+=1
            it+=1
            running_loss=(running_loss*(it-1)+loss.item())/(it)
            if i%1==0:
                print(running_loss)
                print('cur loss: ',loss.item())
        #保存模型，模型名为epoch数+.pth
        final_epoch=epoch==epochs
        if (epoch+1)%1==0:
            model_path=os.path.join(output_dir,str(epoch)+'.pt')
            #记录checkpoints
            ckpt = {'epoch': epoch,
                    'model': ema.ema.module if hasattr(model, 'module') else ema.ema,
                    'optimizer': None if final_epoch else optimizer.state_dict()}
            torch.save(ckpt,model_path)
            torch.save(ckpt,last_path)
            print(model_path)
            #记录log，即每25个epoch的loss
            with open(log_file,'a') as f:
                f.write(str(epoch)+' '+str(running_loss))
                f.write('\n')
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None

if __name__=='__main__':
    args=train_parse_args()
    print(args.model_path)
    #args.model_path = glob.glob('./**/' + args.model_path, recursive=True)  # find file
    #args.data = glob.glob('./**/' + args.data, recursive=True)[0]  # find file
    train_reg(args)