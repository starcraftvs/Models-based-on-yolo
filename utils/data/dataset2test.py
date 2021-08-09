import os
from torch.utils.data import Dataset
#默认文件夹名为label
def Gen_Class_Label(root_dir):
    #建一个classes.txt，用于映射class到label
    class_file=os.path.join(root_dir,'classes.txt')
    #建一个imginfo.txt，记录文件信息，包括路径，种类，及label
    inf_file=os.path.join(root_dir,'imginfo.txt')
    flag=os.path.exists(class_file)
    label=0
    for folder in root_dir:
        label+=1
        class_name=folder
        if not flag:
            with open(class_file,'a') as f:
                f.write(class_name+' '+str(label))
        folder_path=os.path.join(root_dir,folder)
        for img_file in folder_path:
            img_path=os.path.join(img_file)
            with open(inf_file,'a') as f:
                f.write(img_path+' '+label+' '+str(label)




class Class_DataSet(Dataset)：
    def __init__(self,root_dir,transforms=None,data_dir='images',label_dir='label',data_ext='jpg',Norm=True,save_summary=True): #后面两个为是否进行归一化及是否保存大致信息
        #得到图片和label文件的路径
        self.imginfs=GetImgInf(root_dir=root_dir,data_dir=data_dir,label_dir=label_dir,data_ext=data_ext,save_summary=save_summary)
        #Transform
        self.transforms=transforms
        #是否要Normalize数据
        self.Norm=Norm
    def __getitem__(self,index):
        label_path=self.imginfs[index]['label_path']  #得到label地址
        #如果label只有一行，读取label信息
        with open(label_path) as f:
            label=list(map(float, f.readline().split(' ')))
            if self.Norm:
                label=(np.asarray(label) - LOW_VALUE1) / (UP_VALUE1 - LOW_VALUE1) + 1e-8
            img=cv2.imread(self.imginfs[index]['img_path'])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=PIL.Image.fromarray(img)
            if self.transforms is not None:
                img=self.transforms(img)
            label=torch.FloatTensor(label)
        return img,label
    def __len__(self):
        return len(self.imginfs)
