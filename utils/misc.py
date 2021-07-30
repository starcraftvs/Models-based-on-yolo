from typing import Optional ,List
from torch import Tensor
#定义一个用于打包tensors和mask的类
class NestedTensor(object): #object是python定义的新类，有更多可操作对象，更好用
        def __init__(self,tensors,mask: Optional[Tensor]): #表示接受的mask的类型只可以为tensor
            self.tensors=tensors
            self.mask=mask
        #输入进device
        def to(self,device):
            cast_tensor=self.tensors.to(device)
            mask=self.mask
            if mask is not None:
                assert mask is not None #这个就不会执行吧,确认不是None
                cast_mask=mask.to(device)
            else:
                cast_mask=None
            return NestedTensor(cast_tensor,cast_mask)
        def decompose(self):
            return self.tensors, self.mask
        def __repr__(self):
            return str(self.tensors)
