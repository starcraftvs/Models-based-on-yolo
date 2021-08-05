from torchvision import transforms
from utils.data.datasets import Reg_DataSet
from torch.utils.data import DataLoader
from utils.data.transform_factory import create_transform
transforms=create_transform((3,224,224))
import os
# train_dataset=Reg_DataSet(root_dir='correct_images/train',transforms=transforms,data_dir='un_rectify',label_dir='label2')
# train_loader=DataLoader(train_dataset,batch_size=2)
# a1=[]
# a2=[]
# a3=[]
# for batch_idx,(input,target) in enumerate(train_dataset):
#     a1.append(target[0])
#     a2.append(target[1])
#     a3.append(target[2])

# print(len(a1))
# print(max(a1),max(a2),max(a3))
# print(min(a1),min(a2),min(a3))

print(len(os.listdir('correct_images/train/label')))