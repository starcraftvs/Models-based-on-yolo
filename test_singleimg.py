from torchvision import transforms
from utils.data.datasets import Reg_DataSet
from torch.utils.data import DataLoader
from utils.data.transform_factory import create_transform
import torch
import os
import PIL
from constants import *
import argparse

model_dir='/home/gukai/research/Models-based-on-yolo/output/try/5224.pth'
src_path='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train/un_rectify'
dst_path='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images/train/pred_label'