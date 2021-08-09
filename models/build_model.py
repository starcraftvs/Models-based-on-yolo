import argparse

import yaml

from models.experimental import *

#这部分是yolov5改的
class Model(nn.Module):
    def __init__(self, model_cfg='models/CSPDarknetbackbone.yaml', ch=3):
        super(Model, self).__init__()
        if type(model_cfg) is dict:
            self.md = model_cfg  # model dict
        else:  # is *.yaml
            with open(model_cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        self.model, self.save = parse_model(self.md, ch=[ch])  # model, savelist, ch_out
                # Init weights, biases
        torch_utils.initialize_weights(self)
        torch_utils.model_info(self)
    def forward(self, x, augment=False, profile=False):
        if 'head' in self.md:
            if augment:
                img_size = x.shape[-2:]  # height, width
                s = [0.83, 0.67]  # scales
                y = []
                for i, xi in enumerate((x,
                                        torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                                        torch_utils.scale_img(x, s[1]),  # scale
                                        )):
                    # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                    y.append(self.forward_once(xi)[0])

                y[1][..., :4] /= s[0]  # scale
                y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
                y[2][..., :4] /= s[1]  # scale
                return torch.cat(y, 1), None  # augmented inference, train
            else:
                return self.forward_once(x, profile)  # single-scale inference, train
        else:
            return self.forward_once(x,profile)
    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                import thop
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x


def parse_model(md, ch):  # model_dict, input_channels(3)
    print('\n%3s%15s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    gd, gw = md['depth_multiple'], md['width_multiple']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    if 'classify' in md:
        layer_list=md['backbone']+md['classify']
    elif 'head' in md:
        layer_list=md['backbone']+md['head']
    else:
        layer_list=md['backbone']
    for i, (f, n, m, args) in enumerate(layer_list):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, ConvPlus, BottleneckCSP]:
            c1, c2 = ch[f], args[0]

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:
            #c2 = make_divisible(c2 * gw, 8)

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m is BottleneckCSP:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Avgpool:
            args=args
            c2=ch[f]
        elif m is FC:
            c1=ch[f]
            c2=args[0]
            args=[c1,c2]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%15s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        print('.........')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


#这部分是facebook的 detr改的
# """
# Backbone modules.
# """
# from collections import OrderedDict

# import torch
# import torch.nn.functional as F
# import torchvision
# from torch import nn
# from torchvision.models._utils import IntermediateLayerGetter
# from typing import Dict, List

# from utils.misc import NestedTensor, is_main_process

# from .position_encoding import build_position_encoding


# class FrozenBatchNorm2d(torch.nn.Module):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.

#     Copy-paste from torchvision.misc.ops with added eps before rqsrt,
#     without which any other models than torchvision.models.resnet[18,34,50,101]
#     produce nans.
#     """

#     def __init__(self, n):
#         super(FrozenBatchNorm2d, self).__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))

#     def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
#                               missing_keys, unexpected_keys, error_msgs):
#         num_batches_tracked_key = prefix + 'num_batches_tracked'
#         if num_batches_tracked_key in state_dict:
#             del state_dict[num_batches_tracked_key]

#         super(FrozenBatchNorm2d, self)._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict,
#             missing_keys, unexpected_keys, error_msgs)

#     def forward(self, x):
#         # move reshapes to the beginning
#         # to make it fuser-friendly
#         w = self.weight.reshape(1, -1, 1, 1)
#         b = self.bias.reshape(1, -1, 1, 1)
#         rv = self.running_var.reshape(1, -1, 1, 1)
#         rm = self.running_mean.reshape(1, -1, 1, 1)
#         eps = 1e-5
#         scale = w * (rv + eps).rsqrt()
#         bias = b - rm * scale
#         return x * scale + bias

# #用于VisionTransformer的backbone基类
# class BackboneBase(nn.Module):

#     def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
#         #传入的参数的含义：backbone，是否为train_backbone？？是不是自带的？？？
#         super().__init__()
#         for name, parameter in backbone.named_parameters():#这里面输出的name就叫layer几吗？
#             if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
#                 parameter.requires_grad_(False)
#         #是否要返回中间层
#         if return_interm_layers:
#             return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
#         #只返回最终层
#         else:
#             return_layers = {'layer4': "0"}
#         #定义它的body，输入为backbone（应该是个nn.module类），要返回哪些层，之后调用self.body(x)后返回的就是这些层的tensor
#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
#         #估计是输出的通道数
#         self.num_channels = num_channels

#     #跑前向，输入为NestedTensor类
#     def forward(self, tensor_list: NestedTensor):
#         #前向还是只对输入的tensor跑
#         xs = self.body(tensor_list.tensors)
#         #用typing.Dict规定输出的类型为字典
#         out: Dict[str, NestedTensor] = {}
#         #看xs.items里的输出(因为返回了很多layer的输出)
#         for name, x in xs.items():
#             m = tensor_list.mask
#             assert m is not None
#             #根据size对mask上采样，使其和对应层输出的tensor的size一样，并转转换成bool值
#             mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             #打包输出，还是NestedTensor
#             out[name] = NestedTensor(x, mask)
#         return out

# #利用backbonebase的基类写如何backbone
# class Backbone(BackboneBase):
#     """ResNet backbone with frozen BatchNorm."""
    
#     def __init__(self, name: str,
#                  train_backbone: bool,
#                  return_interm_layers: bool,
#                  dilation: bool):
#         #getattr返回对象的属性值，即调的model里各层的name
#         backbone = getattr(torchvision.models, name)(
#             replace_stride_with_dilation=[False, False, dilation],  #用不用空洞卷积代替
#             pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) #是main函数的话就pre_trained，归一化层用FrozenBatchNorm2d
#         num_channels = 512 if name in ('resnet18', 'resnet34') else 2048 #根据选择的网络算num_channels
#         super().__init__(backbone, train_backbone, num_channels, return_interm_layers) #然后就构建了backbone网络


# #把BackBone和PositionEmbedding连在一起
# class Joiner(nn.Sequential): #咋想的呢，nn.Module不就行了
#     def __init__(self,backbone,position_embedding): #牛逼，用nn.Sequential就可以self[0]代表第0层即backbone这么写了
#         super().__init__
#     def forward(self,tensor_list:NestedTensor):
#         xs=self[0](tensor_list) #带着mask过了backbone，但是没用，只是根据各个输出的feature map层构建了大小匹配的mask
#         out: List[NestedTensor]=[]  #规定输出为NestedTensor的列表
#         pos=[] #position的输出
#         for name,x in xs.items():
#             out.append(x)#这个估计是用来记录金子塔输出的
#             #position embedding
#             pos.append(self[1](x).to(x.tensors.dtype)) #positional embedding的结果转换数据类型？？有啥必要？？
#         return out,pos #输出特征图和positional embedding的结果

# #用穿的参搭建backbone
# def build_backbone(args):
#     position_embedding = build_position_encoding(args) #搭建positional embedding
#     train_backbone = args.lr_backbone > 0
#     return_interm_layers = args.masks
#     backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model
