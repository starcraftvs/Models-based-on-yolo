from torch.utils.data import Dataset
import os
import cv2
import PIL
import torch
from constants import *
#from transform_factory import create_transform
#将数据位置及label存入txt文件并返回文件路径
def GetImgInf(root_dir,data_dir='images',label_dir='label',data_ext='jpg',save_summary=True):    #根路径，数据路径，label路径，数据格式，是否保存所有信息在一个文件里
    info=[] #该list用于保存包含images和labels位置的字典
    #如果image和label放一个文件夹
    if data_dir==label_dir:
        data_dir=os.path.join(root_dir,data_dir)
        files=os.listdir(data_dir)
        imgs_path=[os.path.join(data_dir,x) for x in files if x.endswith(data_ext)] #存各个数据的路径
        labels_path=[os.path.join(data_dir,x) for x in files if x.endswith('txt')]  #存各张图片的路径
        for i in range(len(imgs_path)):
            info.append({'img_path':imgs_path[i],'label_path':labels_path[i]})
        return info
    #如果分开放
    else:
        data_dir=os.path.join(root_dir,data_dir)
        label_dir=os.path.join(root_dir,label_dir)
        files_img=os.listdir(data_dir)
        labels_img=os.listdir(label_dir)
        imgs_path=[os.path.join(data_dir,x) for x in files_img if x.endswith(data_ext)]
        labels_path=[os.path.join(label_dir,x) for x in labels_img if x.endswith('txt')]
        for i in range(len(imgs_path)):
            info.append({'img_path':imgs_path[i],'label_path':labels_path[i]})
        return info



class Reg_DataSet(Dataset):
    def __init__(self,root_dir,transforms=None,data_dir='images',label_dir='label',data_ext='jpg',Norm=True,save_summary=True):
        #得到图片和label文件的路径
        self.imginfs=GetImgInf(root_dir,data_dir=data_dir,label_dir=label_dir,data_ext=data_ext,save_summary=save_summary)
        #Transform
        self.transforms=transforms
        #是否要Normalize数据
        self.Norm=Norm
    def __getitem__(self,index):
        label_path=self.imginfs[index]['label_path']  #得到label地址
        #如果label只有一行，读取label信息
        with open(label_path) as f:
            label=list(map(float, f.readline().split(' ')))
            # if self.Norm:
            #     label=(np.asarray(label) - LOW_VALUE) / (UP_VALUE - LOW_VALUE) + 1e-8
            img=cv2.imread(self.imginfs[index]['img_path'])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=PIL.Image.fromarray(img)
            if self.transforms is not None:
                img=self.transforms(img)
            label=torch.FloatTensor(label)
        return img,label
    def __len__(self):
        return len(self.imginfs)

