from torchvision import transforms
from utils.data.datasets import Reg_DataSet
from torch.utils.data import DataLoader
from utils.data.transform_factory import create_transform
import torch
import os
import PIL
from constants import *
import argparse
import torch.nn as nn
from models.build_model import Model
from utils.defaults import test_parse_args
from utils import torch_utils

def test_reg(args):
    #读取参数
    src_dir=args.test_dir
    dst_dir=args.output_dir
    label_dir=args.label_dir
    #输出文件夹
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    #图片size转换
    transforms=create_transform(args.input_size)
    #验证的device
    device = torch_utils.select_device(args.device, apex=False, batch_size=1)
    #load模型
    model=Model(args.model_path)
    model.to(device)
    ckpt=torch.load(args.weights_path,map_location=device)
    ckpt['model'] = \
        {k: v for k, v in ckpt['model'].state_dict().items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
    #如果要验证，要算loss
    if args.val:
        loss_fn=nn.L1Loss()
        label_dir=args.label_dir
    for img_file in os.listdir(src_dir):
        img_path=os.path.join(src_dir,img_file)
        img=transforms(PIL.Image.open(img_path))
        img=img.unsqueeze(0)
        img=img.to(device)
        pred=model(img)
        label_path=os.path.join(label_dir,img_file[:-3]+'txt')
        if args.val:
            with open (label_path,'r') as f:
                label=list(map(float, f.readline().strip().split(' ')))
            label=(label-LOW_VALUE1)/(UP_VALUE1-LOW_VALUE1)
            label=torch.FloatTensor(label).to(device).unsqueeze(0)
            loss=loss_fn(pred,label)
            print(img_file)
            print(label)
            print(pred)
            print(loss)
            pred=pred.cpu().detach().numpy().reshape(1,4)
            pred=(pred*(UP_VALUE1-LOW_VALUE1))+LOW_VALUE1
        with open(os.path.join(dst_dir,img_file[:-3]+'txt'),'w') as f:
            f.write(str(' '.join(str(x) for x in pred[0])))
            if args.val:
                f.write('\n')
                f.write(str(loss.item()))
        print(img_path)

if __name__=='__main__':
    args=test_parse_args()
    #args.model_path = glob.glob('./**/' + args.model_path, recursive=True)  # find file
    #args.data = glob.glob('./**/' + args.data, recursive=True)[0]  # find file
    test_reg(args)