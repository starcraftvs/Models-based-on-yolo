import argparse
import torch
from torchvision import transforms
from utils.data.datasets import Reg_DataSet
from torch.utils.data import DataLoader
from utils.data.transform_factory import create_transform
transforms=create_transform((3,224,224))
from models.build_model import Model
from utils.data.classify_datasets import *
from utils.utils import *
from utils.data.dataloader import *
import numpy as np
from utils.data.transform_factory import *
from utils.utils import *
import yaml
import torch.nn as nn
#训练超参,目前只用lr
hyp = {'lr0': 0.0001,  # initial learning rate (SGD=1E-2, Adam=1E-3)
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


#获取训练参数
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
#有没有config文件
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
#input_size
parser.add_argument('--input_size', type=tuple, default=[488,488],
                    help='input_size default(3,244,244)')
#batch_size
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size')
#训练路径
parser.add_argument('--train_dir', type=str, default='/home/gukai/research/Models-based-on-yolo/correct_images/train',
                    help='train_data dir')
#验证路径
parser.add_argument('--val_dir', type=str, default='/fast_data2/India/TranData/RussiaBiscuits/RussiaBiscuits_wft_0715',
                    help='val_dir')
#训练epochs
parser.add_argument('--epochs', type=int, default=10000,
                    help='epochs')
#模型文件
parser.add_argument('--config_file', type=str,default='models/MobileNet.yaml',
                    help='model config-file')

#输出地址
parser.add_argument('--output_dir', type=str,default='output/210805',
                    help='output_dir to save model')

#是否继续上次训练
parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text




def train_reg(hyp):
    #获取参数
    args, args_text = _parse_args()
    epochs=args.epochs
    output_dir=args.output_dir
    epochs = args.epochs  # 300
    batch_size = args.batch_size  # 64
    #设置随机种子数，使得随机化的数都一样，每次训练可复现，而且不设置为0的话会自动搜索最合适的计算方法，加速计算
    init_seeds(1)
    #获取数据，并transform
    transforms=create_transform(args.input_size)
    train_dataset=Reg_DataSet(root_dir=args.train_dir,transforms=transforms,data_dir='un_rectify',label_dir='label2')
    #回归的vector长度
    n_v=3
    #load数据
    train_loader=DataLoader(train_dataset,batch_size=batch_size)
    #删除之前结果

    #创建模型
    model=Model(args.config_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #数据量
    last_idx=len(train_dataset)-1
    #loss function
    loss_fn=nn.SmoothL1Loss()
    #优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=hyp['lr0'])
    #设置保存模型及log的路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file=os.path.join(output_dir,'log.txt')
    #要不要从保存处重跑
    if args.resume==True:
        #如果要，第几个epoch
        with open(log_file,'r') as f:
            last_epoch=int(f.readlines()[-1].strip().split(' ')[0])
        start_epoch=last_epoch+1
        #load 该epoch的模型
        model_path=os.path.join(output_dir,str(last_epoch)+'.pth')
        model=torch.load(model_path)
        model.to(device)
        #如果不要，重新create模型
    else:
        start_epoch=0
        model=Model(args.config_file)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
    #跑多少个epoch
    for epoch in range(start_epoch,epochs):
        i=0
        for batch_idx, (input, target) in enumerate(train_loader):
            last_batch = batch_idx == last_idx
            #target=torch.eye(num_classes)[target,:]
            #print(np.array(input).shape)
            input=input.to(device)
            target=target.to(device)
            predict=model(input)
            #print(predict.shape)
            #print(target)
            loss=loss_fn(predict,target)
            loss.backward()
            optimizer.step()

            if i%100==0:
                print(loss)
            i+=1
        #保存模型，模型名为epoch数+.pth
        if (epoch+1)%25==0:
            model_path=os.path.join(output_dir,str(epoch)+'.pth')
            torch.save(model,model_path)
            print(model_path)
            #记录log，即每25个epoch的loss
            with open(log_file,'a') as f:
                f.write(str(epoch)+' '+str(loss.item))
                f.write('\n')

train_reg(hyp)
# transform=create_transform([224,224])
# img=PIL.Image.open('/home/gukai/docker/TestNewEnviornment/cardboardtest/tests/images/10300079.jpg')
# img=transform(img)
# np_img=np.array(img)
# print(np_img.shape)