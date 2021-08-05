#transform的组件

import torch
import numpy as np
import PIL
class ToNumpy:

    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        return np_img

#转换为Tensor
class ToTensor:
    #要转换成的数据格式
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype
    #转化成tensor
    def __call__(self, img):
        #pil图转array
        if isinstance(img,PIL.Image.Image):
            #np_img = np.array(img)
            np_img = np.array(img).astype('uint8')
        #opencv读的转通道,在dataset类里已经转了
        else:
            np_img=img
        #     if img.ndim>=3:
        #         np_img = img[...,::-1]
        #灰度图加维度
        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        #通道转换
        np_img = np.rollaxis(np_img, 2)  # HWC to CHW
        
        return torch.from_numpy(np_img).to(dtype=self.dtype)