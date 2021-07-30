import torch.nn as nn
import torch
import torch.functional as F
import numpy as np
import math
from utils.misc import NestedTensor




#二维positional embedding
class PositionEmbeddingSine(nn.Module):
    def __init__(self,num_pos_feats=256,temperature=10000,normalize=False,scale=None):
        super().__init__()
        self.num_pos_feats=num_pos_feats #输入的d_k维度的一半，因为一半加sin，一半加cos，按cspdarknet是512/2=256
        self.temperature=temperature #sqrt(d_k)
        self.normalize=normalize #是否要normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed") #定了scale就必须要normalize，不然白定了
        #需要normalize但没有定scale的情况下，算下scale
        if scale is None:
            scale=2*math.pi
        self.scale=scale
    #前向
    def forward(self,tensor_list:NestedTensor): #输入是个b,c,w,h的tensor和mask打包好的NestedTensor类
        x=tensor_list.tensors #b,c,w,h维
        mask=tensor_list.mask   #输入打包的mask，默认为None
        assert mask is not None #mask不能为None，为None则报错
        not_mask=~mask #按位取反  没看懂，回头看下
        #二维position embedding计算
        #x方向，后面调用的时候算下来是1 1 1 1 ..  2 2 2 2... 3 3 3...
        y_embed=not_mask.cumsum(1,dtype=torch.float32) 
        #y方向，后面调用的时候算下来是1 2 3 4 ... 1 2 3 4...
        x_embed=not_mask.cumsum(2,dtype=torch.float32) #x.cumsum是用来某个维度求累加和的
        #要不要归一化
        if self.normalize: 
            eps=1e-6    #防止分母为1
            y_embed=y_embed/(y_embed[:,-1:,:]+eps)*self.scale    #先化成角度,最大的数为2pi，一个周期
            x_embed=y_embed/(y_embed[:,:,-1:]+eps)*self.scale   

        dim_t=torch.arange(self.num_pos_feats,dtype=torch.float32,device=x.device)  #这儿用range不就行了？0:255
        dim_t=self.temperature**(2*(dim_t//2))/self.num_pos_feats #见公式//表示向下取整，但仍是float数

        #输出shape=b,h,w,256（原channel为512
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t  #维度上变成了b,H,W,256
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) #一半维度(128)sin，一半cos，维度上b,H,W,256切片变成俩b,H,W,128，分别cos,sin，再stack变成b,H,W,2,64，再摊平变成b,H,W,128，感觉不如直接cat啊
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) #合并后把维度变成b,n=512,h,w
    # 每个特征图的xy位置都编码成512的向量，其中前256是y方向编码，而256是x方向编码
        return pos
    # b,n=256,h,w

#利用参数构建positional embedding类构建positioanl_encoding
def build_position_encoding(args):
    N_steps=args.hidden_dim//2 #就是上面的num_pos_feats，就是一半sin，一半cos的数量，输入维度的一半
    #position_embedding的类型，是encoder那儿的还是decoder那儿的
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    #encoder那儿输入的positionembedding，不过，这个不是N_steps是固定的嘛？？？
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding