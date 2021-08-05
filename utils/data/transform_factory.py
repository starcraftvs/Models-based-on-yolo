import math

import torch
from torchvision import transforms
from .transform import ToTensor
#transform 图片并进行归一化（感觉不需要）
def transforms_noaug_train(
        img_size=[224,224]   #img_size
        # interpolation='bilinear',   #resize的时候的插值方法
):
    # if interpolation == 'random':
    #     # random interpolation not supported with no-aug
    #     interpolation = 'bilinear'
    #resize加裁剪????
    tfl = [
        transforms.Resize(img_size),
        # transforms.Resize(img_size, _pil_interp(interpolation)),
        transforms.CenterCrop(img_size)
    ]
    #转换成Tensor
    tfl += [
        ToTensor(),
    #     transforms.Normalize(
    #         mean=torch.tensor(mean),
    #         std=torch.tensor(std))
    ]
    return transforms.Compose(tfl)

def create_transform(input_size):
    #看输入的size是三维tuple还是二维h,w的list
    if isinstance(input_size, tuple):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    transform=transforms_noaug_train(img_size)
    print()
    return transform