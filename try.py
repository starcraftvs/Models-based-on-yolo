import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.utils.tensorboard import SummaryWriter
import cv2
import test  # import test.py to get mAP after each epoch
from models.build_model import Model
from utils.data.classify_datasets import *
from utils.utils import *
from models.transformer import VisionTransformer
from utils.data.dataloader import *
import numpy as np
# img=cv2.resize(cv2.imread('datasets/new1/fffecef2-b57c-40dd-941b-1d342af23686_5_DFbuwrlX0G3uC1vzOykYg_1.jpg'),(640,640))
# print(img.shape)
# img=torch.FloatTensor(img)
# img=img.transpose(0,2)
# img=torch.unsqueeze(img,0)
m1=Model('models/MobileNet.yaml')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
m1.to(device)
# img=m1(img)
# print(img.shape)
# print(img)

# model_cfg='models/MobileNet.yaml'
# with open(model_cfg) as f:
#     md = yaml.load(f, Loader=yaml.FullLoader)  # model dict

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

#获取参数
args, args_text = _parse_args()
train_dir='/fast_data2/India/TranData/RussiaBiscuits/RussiaBiscuits_wft_0715'
#获取数据
dataset_train = Dataset(train_dir, balance=False)
#种类数
num_classes=(len(dataset_train.class_to_idx))
#load数据
loader_train = create_loader(
    dataset_train,
    input_size=args.input_size,
    batch_size=args.batch_size,
    is_training=True,
)


#数据量
last_idx=len(dataset_train)-1
loss_fn=nn.CrossEntropyLoss()
# i=0
# for batch_idx, (input, target) in enumerate(dataset_train):
#     if i==20000:
#         print(target)
#         break
#     i+=1
i=0
optimizer = torch.optim.Adam(m1.parameters(), lr=0.0001)
for batch_idx, (input, target) in enumerate(loader_train):
    last_batch = batch_idx == last_idx
    #target=torch.eye(num_classes)[target,:]
    #print(np.array(input).shape)
    input,target=input.to(device),target.to(device)
    predict=m1(input)
    #print(predict.shape)
    #print(target)
    loss=loss_fn(predict,target)
    loss.backward()
    optimizer.step()

    if i%100==0:
        print(loss)
    i+=1

