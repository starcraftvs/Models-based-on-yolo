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
from models.backbone import CSPDarknetBackbone
from utils.datasets import *
from utils.utils import *
from models.transformer import VisionTransformer

img=cv2.resize(cv2.imread('datasets/new1/fffecef2-b57c-40dd-941b-1d342af23686_5_DFbuwrlX0G3uC1vzOykYg_1.jpg'),(640,640))
img=torch.FloatTensor(img)
img=img.transpose(0,2)
img=torch.unsqueeze(img,0)
m1=CSPDarknetBackbone()
img1=m1(img)
img2=m1(img)
print(img1.shape)
# img1=img1.contiguous().view(1,400,512)
# m2=VisionTransformer(512,512,0,0)
# img1=m2(img1,img1)
# print(img1.shape)