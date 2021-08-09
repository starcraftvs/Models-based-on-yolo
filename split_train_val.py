import os
import shutil
import math
root='/home/gukai/research/Models-based-on-yolo/datafolder/correct_images'
os.chdir(root)
folder1s=['train','val']
folder2s=['un_rectify','rectify','label','label3']
paths=[]
for folder1 in folder1s:
    if not os.path.exists(folder1):
        os.mkdir(folder1)
    for folder2 in folder2s:
        paths.append(os.path.join(folder1,folder2))
        if not os.path.exists(paths[-1]):
            os.mkdir(paths[-1])
unrectify='unrectify/'
rectify='rectify/'
fs=os.listdir(unrectify)
num_d=len(fs)
j=int(num_d/4)*3
for i in range(j):
    f=fs[i]
    rectify_f=os.path.join(rectify,f)
    unrectify_f=os.path.join(unrectify,f)
    label=os.path.join(rectify,f[:-3]+'txt')
    with open (label,'r') as fil:
        label_data=fil.readline()
        label_data2=list(label_data.split(' '))
        #del label_data2[1]
        label_data2[0]=str(math.sin(float(label_data2[0])/180*math.pi))
        label_data2[1]=str(math.exp(float(label_data2[1])))
        label_data2[2]=str(math.exp(float(label_data2[2])))
    if os.path.exists(rectify_f):
        shutil.copyfile(unrectify_f,os.path.join(paths[0],f))
        shutil.copyfile(rectify_f,os.path.join(paths[1],f))
        shutil.copyfile(label,os.path.join(paths[2],f[:-3]+'txt'))
        with open(os.path.join(paths[3],f[:-3]+'txt'),'w') as fil:
            fil.write(' '.join(label_data2))

for i in range(j,num_d):
    f=fs[i]
    rectify_f=os.path.join(rectify,f)
    unrectify_f=os.path.join(unrectify,f)
    label=os.path.join(rectify,f[:-3]+'txt')
    with open (label,'r') as fil:
        label_data=fil.readline()
        label_data2=list(label_data.split(' '))
        #del label_data2[1]
        label_data2[0]=str(math.sin(float(label_data2[0])/180*math.pi))
        label_data2[1]=str(math.exp(float(label_data2[1])))
        label_data2[2]=str(math.exp(float(label_data2[2])))
        
    if os.path.exists(rectify_f):
        print(os.path.join(paths[3],f))
        shutil.copyfile(unrectify_f,os.path.join(paths[4],f))
        shutil.copyfile(rectify_f,os.path.join(paths[5],f))
        shutil.copyfile(label,os.path.join(paths[6],f[:-3]+'txt'))
        with open(os.path.join(paths[7],f[:-3]+'txt'),'w') as fil:
            fil.write(' '.join(label_data2))