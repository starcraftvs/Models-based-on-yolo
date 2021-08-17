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
from utils.engine import launch
from utils.defaults import _parse_args
from torch.nn.parallel import DistributedDataParallel
from utils import comm

def train_reg(args):
    #训练超参,目前只用lr
    hyp = {'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
        #    'momentum': 0.937,  # SGD momentum
        #    'weight_decay': 5e-4,  # optimizer weight decay
        #    'giou': 0.05,  # giou loss gain
        #    'cls': 0.58,  # cls loss gain
        #    'cls_pw': 1.0,  # cls BCELoss positive_weight
        #    'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
        #    'obj_pw': 1.0,  # obj BCELoss positive_weight
        #    'iou_t': 0.20,  # iou training threshold
        #    'anchor_t': 4.0,  # anchor-multiple threshold
        #    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
        #    'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
        #    'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
        #    'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
        #    'degrees': 0.0,  # image rotation (+/- deg)
        #    'translate': 0.0,  # image translation (+/- fraction)
        #    'scale': 0.5,  # image scale (+/- gain)
        #    'shear': 0.0 # image shear (+/- deg)
        }  
    # 如果有hyp*.txt文件，就Overwrite hyp with hyp*.txt (optional)
    f = glob.glob('hyp*.txt')
    if f:
        print('Using %s' % f[0])
        for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
            hyp[k] = v
    print(hyp)
    #获取参数
    args = _parse_args()
    epochs=args.epochs
    output_dir=args.output_dir
    epochs = args.epochs  # 300
    batch_size = args.batch_size  # 64
    #设置随机种子数，使得随机化的数都一样，每次训练可复现，而且不设置为0的话会自动搜索最合适的计算方法，加速计算
    init_seeds(1)
    #获取数据，并transform
    transforms=create_transform(args.input_size)
    train_dataset=Reg_DataSet(root_dir=args.train_dir,transforms=transforms,data_dir=args.data_dir,label_dir=args.label_dir)

    #load数据
    train_loader=DataLoader(train_dataset,batch_size=batch_size)

    #数据量
    last_idx=len(train_dataset)-1

    #设置保存模型及log的路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file=os.path.join(output_dir,'log.txt')
    #建立模型
    model=Model(args.model_path)
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )     
    #loss function
    loss_fn=nn.SmoothL1Loss()
    #优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=hyp['lr0'])
    #跑多少个epoch
    i=0 #记录跑了多少iteration
    start_epoch=0
    for epoch in range(start_epoch,epochs):
        running_loss=0
        #每个epoch里跑了多少个epoch
        it=0
        for batch_idx, (input, target) in enumerate(train_loader):
            last_batch = batch_idx == last_idx
            #target=torch.eye(num_classes)[target,:]
            #print(np.array(input).shape)
            predict=model(input)
            #print(predict.shape)
            #print(target)
            loss=loss_fn(predict,target)
            loss.backward()
            optimizer.step()
            i+=1
            it+=1
            running_loss=(running_loss*(it-1)+loss.item())/(it)
            if i%100==0:
                print(running_loss)
                print('cur loss: ',loss.item())
        #保存模型，模型名为epoch数+.pth
        if (epoch+1)%25==0:
            model_path=os.path.join(output_dir,str(epoch)+'.pth')
            torch.save(model,model_path)
            print(model_path)
            #记录log，即每25个epoch的loss
            with open(log_file,'a') as f:
                f.write(str(epoch)+' '+str(running_loss))
                f.write('\n')

if __name__=='__main__':
    args=_parse_args()
    print()
    print("Command Line Args:", args)
    launch(
        train_reg,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )