from torchvision import transforms
from utils.data.datasets import Reg_DataSet
from torch.utils.data import DataLoader
from utils.data.transform_factory import create_transform
import torch
import os
import PIL
from constants import *
transforms=create_transform((3,488,488))
model_dir='/home/gukai/research/Models-based-on-yolo/output/210805/149.pth'
src_path='/home/gukai/research/Models-based-on-yolo/correct_images/train/un_rectify'
dst_path='/home/gukai/research/Models-based-on-yolo/correct_images/train/pred_label'
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
    pred=pred.cpu().detach().numpy()
    pred=pred*(UP_VALUE1-LOW_VALUE1)+LOW_VALUE1
    with open(os.path.join(dst_path,img_file[:-3]+'txt'),'w') as f:
        f.write(str(pred))
    print(img_path)