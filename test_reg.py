from torchvision import transforms
from utils.data.datasets import Reg_DataSet
from torch.utils.data import DataLoader
from utils.data.transform_factory import create_transform
import torch
import os
import PIL
from constants import *
import argparse
#获取训练参数
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
#有没有config文件
# parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
#                     help='YAML config file specifying default arguments')
#input_size
parser.add_argument('--input_size', type=tuple, default=[488,488],
                    help='input_size default(3,244,244)')
#batch_size
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size')
#训练路径
parser.add_argument('--train_dir', type=str, default='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train',
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
parser.add_argument('--resume', action='store_true',default=False,
                    help='whether to resume')
transforms=create_transform((3,488,488))
model_dir='/home/gukai/research/Models-based-on-yolo/output/210809/99.pth'
src_path='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train/un_rectify'
dst_path='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train/pred_label'
if not os.path.exists(dst_path):
    os.makedirs(dst_path)
model=torch.load(model_dir)
for img_file in os.listdir(src_path):
    img_path=os.path.join(src_path,img_file)
    img=transforms(PIL.Image.open(img_path))
    img=img.unsqueeze(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img=img.to(device)
    pred=model(img)
    pred=pred.cpu().detach().numpy().reshape(1,4)
    pred=pred*(UP_VALUE1-LOW_VALUE1)+LOW_VALUE1
    with open(os.path.join(dst_path,img_file[:-3]+'txt'),'w') as f:
        f.write(str(pred))
    print(img_path)