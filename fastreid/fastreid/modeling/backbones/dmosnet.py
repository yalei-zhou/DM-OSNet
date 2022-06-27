# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on:
# https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py

# import imp
import logging
# from OpCounterTransformer.thop import profile
import torch
from torch import nn
# from fvcore.nn import FlopCountAnalysis,parameter_count_table,parameter_count
# import fvcore
# import torchreid
import collections
from torchstat import stat

# import os
# import sys
# print('当前工作路径: ', os.getcwd())


# print('导包路径为: ')
# sys.path.append("/home/zyl/fast-reid")
# sys.path.append("/home/zyl/fast-reid")
# for p in sys.path:
#     print(p)

# from ...layers import get_norm
# from /home/zyl/fast-reid/fastreid/layers import 
# import sys


# import cuml
from fastreid.layers import get_norm
from fastreid.utils import comm
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
# sys.path.append("./")
# from fastreid.modeling.backbones.build import BACKBONE_REGISTRY
from .build import BACKBONE_REGISTRY



logger = logging.getLogger(__name__)
model_urls = {
    'osnet_x1_0':
        'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',
    'osnet_x0_75':
        'https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq',
    'osnet_x0_5':
        'https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i',
    'osnet_x0_25':
        'https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs',
    'osnet_ibn_x1_0':
        'https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l'
}


class Attention(nn.Module):
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, drop=0., norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size

        self.norm = norm_layer(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)#dim = 64
        self.proj = nn.Conv2d(dim, dim, 1)
        self.drop = nn.Dropout2d(drop, inplace=True)
        # self.conv2 = nn.Conv2d(
        #     dim,
        #     dim,
        #     3,
        #     stride=1,
        #     padding=1,
        #     bias=False,
        #     groups=dim
        # )
        if grid_size > 1:
            self.grid_norm = norm_layer(dim)
            self.avg_pool = nn.AvgPool2d(ds_ratio, stride=ds_ratio)
            self.ds_norm = norm_layer(dim)
            self.q = nn.Conv2d(dim, dim, 1)
            self.kv = nn.Conv2d(dim, dim * 2, 1)
        # self.transform_conv = nn.Conv2d(self.num_heads,self.num_heads,kernel_size=1,stride=1)
        # self.transform_norm = nn.InstanceNorm2d(self.num_heads)
        # self.rel_h = nn.Parameter(torch.randn([1, self.num_heads, self.head_dim, 1,64 ]), requires_grad=True)
        # self.rel_w = nn.Parameter(torch.randn([1, self.num_heads, self.head_dim, 32, 1]), requires_grad=True)
        # self.rel_h = nn.Parameter(torch.randn([1, dim, 1,64 ]), requires_grad=True)
        # self.rel_w = nn.Parameter(torch.randn([1, dim, 32, 1]), requires_grad=True)
    def forward(self, x):
        B, C, H, W = x.shape#16,64,64,32
        qkv = self.qkv(self.norm(x))#64,192,64,32

        if self.grid_size > 1:
            grid_h, grid_w = H // self.grid_size, W // self.grid_size#(8,4)
            qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, grid_h,#(64,3,1,64,8,8,4,8)
                self.grid_size, grid_w, self.grid_size) # B QKV Heads Dim H GSize W GSize
            qkv = qkv.permute(1, 0, 2, 4, 6, 5, 7, 3)#permute将tensor的维度换位。(3,64,1,8,4,8,8,64)
            qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size, self.head_dim)#(3,2048,64,64)
            q, k, v = qkv[0], qkv[1], qkv[2]#2048,64,64

            attn = (q @ k.transpose(-2, -1)) * self.scale#transpose交换一个tensor的两个维度.2048,64,64
            attn = attn.softmax(dim=-1)
            grid_x = (attn @ v).reshape(B, self.num_heads, grid_h, grid_w,#@是用来对tensor进行矩阵相乘的
                self.grid_size, self.grid_size, self.head_dim)#(64,1,8,4,8,8,64)
            grid_x = grid_x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, C, H, W)#(64,64,64,32)
            grid_x = self.grid_norm(x + grid_x)

            q = self.q(grid_x).reshape(B, self.num_heads, self.head_dim, -1)#(64,1,64,2048)
            q = q.transpose(-2, -1)#(64,1,2048,64)
            kv = self.kv(self.ds_norm(self.avg_pool(grid_x)))#64,128,8,4
            kv = kv.reshape(B, 2, self.num_heads, self.head_dim, -1)#64,2,1,64,32
            kv = kv.permute(1, 0, 2, 4, 3)#2,64,1,32,64
            k, v = kv[0], kv[1]#64,1,32,64
        else:
            qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, -1)
            qkv = qkv.permute(1, 0, 2, 4, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
        # s = k.transpose(-2, -1)
        # aa = q@s
        # bb = aa*self.scale
        attn = (q @ k.transpose(-2, -1)) * self.scale#64,1,2048,32
        
        # pos = self.ds_norm(self.avg_pool((self.rel_h + self.rel_w)))
        # # pos = pos.reshape(1, 2, self.num_heads, self.head_dim, -1)
        # content_position = pos.view(1, self.num_heads,self.head_dim, -1).permute(0, 1, 3, 2)
        # content_position = torch.matmul(q,content_position.transpose(-2, -1))
        # attn = attn+content_position
        # attn = self.transform_conv(attn)
        attn = attn.softmax(dim=-1)
        # attn = self.transform_norm(attn)
        global_x = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)#64,64,64,32
        if self.grid_size > 1:
            global_x = global_x + grid_x
        x = self.drop(self.proj(global_x))#64,64,64,32

        return x


##########
# Basic layers
##########
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            bn_norm,
            stride=1,
            padding=0,
            groups=1,
            IN=False
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups
        )
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = get_norm(bn_norm, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, bn_norm, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups
        )
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution.
    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels, bn_norm):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels
        )
        self.bn = get_norm(bn_norm, out_channels)
        # self.bn = bn_norm

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class attnConv3x3(nn.Module):
    """Lightweight 3x3 convolution.
    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels, bn_norm,grid_size,ds_ratio,drop):
        super(attnConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.attn2 = Attention(out_channels,32,grid_size,ds_ratio,drop)
        self.bn = get_norm(bn_norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attn2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

##########
# Building blocks for omni-scale feature learning
##########
class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
            self,
            in_channels,
            num_gates=None,
            return_gates=False,
            gate_activation='sigmoid',
            reduction=16,
            layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None: num_gates = in_channels
        self.return_gates = return_gates

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm: self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = nn.Identity()
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None: x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.gate_activation(x)
        if self.return_gates: return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block."""

    def __init__(
            self,
            in_channels,
            out_channels,
            bn_norm,
            IN=False,
            bottleneck_reduction=4,
            mhsaflag = False ,
            mhsalist = [0,0,0,0],
            gridflag = True,
            grid = [4,4,0],
            **kwargs
    ):
        super(OSBlock, self).__init__()
        heads = 4
        mid_channels = out_channels // bottleneck_reduction
        self.mhsaflag = mhsaflag
        self.conv1 = Conv1x1(in_channels, mid_channels, bn_norm)
        
        self.conv2a = LightConv3x3(mid_channels, mid_channels, bn_norm)
        # self.conv2a = nn.Sequential()

        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            # Attention(mid_channels,32,8,8,0)
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            LightConv3x3(mid_channels, mid_channels, bn_norm),
            # Attention(mid_channels,32,8,8,0)
        )
        if gridflag:
            # self.conv2d = nn.Sequential()
            self.conv2d = nn.Sequential(
            attnConv3x3(mid_channels, mid_channels, bn_norm,grid[0],grid[1],grid[2])

            # Attention(mid_channels,32,8,8,0)
        )
        else:

            self.conv2d = nn.Sequential(
                LightConv3x3(mid_channels, mid_channels, bn_norm),
                LightConv3x3(mid_channels, mid_channels, bn_norm),
                LightConv3x3(mid_channels, mid_channels, bn_norm),
                LightConv3x3(mid_channels, mid_channels, bn_norm),
                # Attention(mid_channels,32,8,8,0)
            )
            

        self.attn = None
        if mhsaflag:

            self.attn = Attention(mid_channels,32,8,8,0)
        
        # conv2list = (self.conv2a,self.conv2b,self.conv2c,self.conv2d)
        # if mhsaflag:
        #     for i,theconv in enumerate(conv2list):
        #         if mhsalist[i]:
        #             theconv.add_module(f"{i}",MHSA(mid_channels, width=16, height=8, heads=heads))
        #         else:
        #             theconv.add_module(f"{i}",LightConv3x3(mid_channels, mid_channels, bn_norm))
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels, bn_norm)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels, bn_norm)
        self.IN = None


        if IN: self.IN = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        # x2e = self.conv2e(x1)
        # x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)+self.gate(x2e)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) +self.gate(x2d)

        if self.attn is not None:

            x2 = self.attn(x2)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        # print("##########")
        if self.IN is not None:

            out = self.IN(out)
        return self.relu(out)


##########
# Network architecture
##########
class OSNet(nn.Module):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(
            self,
            blocks,
            layers,
            channels,
            bn_norm,
            IN=True,
            # size = [256,128]#16,8
            **kwargs
    ):
        super(OSNet, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1

        # convolutional backbone
        self.conv1 = ConvLayer(3, channels[0], 7, bn_norm, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            bn_norm,
            reduce_spatial_size=True,
            IN=IN,
            gridflag = True,
            grid = [8,8,0]
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            bn_norm,
            reduce_spatial_size=True,
            gridflag = False,
            grid = [8,4,0]
            
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            bn_norm,
            reduce_spatial_size=False,
            
            mhsaflag  = False,
            mhsalist = [1,1,1,1],
            gridflag = False,
            grid = [8,2,0]

        )
        self.conv5 = Conv1x1(channels[3], channels[3], bn_norm)

        self._init_params()

    def _make_layer(
            self,
            block,
            layer,
            in_channels,
            out_channels,
            bn_norm,
            reduce_spatial_size,
            IN=False,
            mhsaflag  = False,
            mhsalist = [0,0,0,0],
            gridflag = True,
            grid = [4,4,0]
    ):
        layers = []

        layers.append(block(in_channels, out_channels, bn_norm, IN=IN,mhsaflag=mhsaflag, mhsalist=mhsalist,gridflag = gridflag,grid=grid))
        for i in range(1, layer):
            layers.append(block(out_channels, out_channels, bn_norm, IN=IN, mhsaflag=mhsaflag, mhsalist=mhsalist,gridflag = gridflag,grid = grid))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels, bn_norm),
                    nn.AvgPool2d(2, stride=2),
                )
            )

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


def init_pretrained_weights(model, key=''):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown
    from collections import OrderedDict
    import warnings
    import logging

    logger = logging.getLogger(__name__)

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    filename = key + '_imagenet.pth'
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        logger.info(f"Pretrain model don't exist, downloading from {model_urls[key]}")
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    return model_dict

def sanchuzichuan(ss1,ss2):
    ss1list = ss1.split(".")
    ss = ""
    falg = False
    for i,s in enumerate(ss1list):
        if "".join(ss1list[i]) == ss2:
            falg = True
            break
    if falg:
        del ss1list[i]
        del ss1list[i]
    ss = ".".join(ss1list)
    print(ss)
    return ss
        

    
            
@BACKBONE_REGISTRY.register()
def build_dmosnet_backbone(cfg):
    """
    Create a OSNet instance from config.
    Returns:
        OSNet: a :class:`OSNet` instance
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # hw = cfg.INPUT.SIZE_TRAIN
    # fmt: on

    # pretrain      = False
    # pretrain_path = "/home/zyl/fast-reid/weights/LUPersonpretrain/checkpoint.pth"
    # with_ibn      = True
    # bn_norm       = "BN"
    # depth         = "x1_0"
    # hw = cfg.INPUT.SIZE_TRAIN
    # fmt: on

    num_blocks_per_stage = [2, 2, 2]
    num_channels_per_stage = {
        "x1_0": [64, 256, 384, 512],
        "x0_75": [48, 192, 288, 384],
        "x0_5": [32, 128, 192, 256],
        "x0_25": [16, 64, 96, 128]}[depth]
    model = OSNet([OSBlock, OSBlock, OSBlock], num_blocks_per_stage, num_channels_per_stage,
                  bn_norm, IN=with_ibn)

    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                # sanchuzichuan("module.dsbn.1.j","dsbn")
                state_dict = collections.OrderedDict([(k[16:], v) if k.startswith('module.backbone') else (k, v) for k, v in state_dict["state_dict"].items()])
                state_dict = collections.OrderedDict([(sanchuzichuan(k,"dsbn"), v) for k, v in state_dict.items()])
                # state_dict["state_dict"]
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            if with_ibn:
                pretrain_key = "osnet_ibn_" + depth
            else:
                pretrain_key = "osnet_" + depth

            state_dict = init_pretrained_weights(model, pretrain_key)

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )
    return model


# def build_osnet_backbone():
#     """
#     Create a OSNet instance from config.
#     Returns:
#         OSNet: a :class:`OSNet` instance
#     """

#     # fmt: off
#     pretrain      = False
#     pretrain_path = "/home/zyl/fast-reid/weights/model_best101.pth"
#     with_ibn      = True
#     bn_norm       = "BN"
#     depth         = "x1_0"
#     # hw = cfg.INPUT.SIZE_TRAIN
#     # fmt: on

#     num_blocks_per_stage = [2, 2, 2]
#     num_channels_per_stage = {
#         "x1_0": [64, 256, 384, 512],
#         "x0_75": [48, 192, 288, 384],
#         "x0_5": [32, 128, 192, 256],
#         "x0_25": [16, 64, 96, 128]}[depth]
#     model = OSNet([OSBlock, OSBlock, OSBlock], num_blocks_per_stage, num_channels_per_stage,
#                   bn_norm, IN=with_ibn)

#     if pretrain:
#         # Load pretrain path if specifically
#         if pretrain_path:
#             try:
#                 state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
#                 logger.info(f"Loading pretrained model from {pretrain_path}")
#             except FileNotFoundError as e:
#                 logger.info(f'{pretrain_path} is not found! Please check this path.')
#                 raise e
#             except KeyError as e:
#                 logger.info("State dict keys error! Please check the state dict.")
#                 raise e
#         else:
#             if with_ibn:
#                 pretrain_key = "osnet_ibn_" + depth
#             else:
#                 pretrain_key = "osnet_" + depth

#             state_dict = init_pretrained_weights(model, pretrain_key)

#         incompatible = model.load_state_dict(state_dict, strict=False)
#         if incompatible.missing_keys:
#             logger.info(
#                 get_missing_parameters_message(incompatible.missing_keys)
#             )
#         if incompatible.unexpected_keys:
#             logger.info(
#                 get_unexpected_parameters_message(incompatible.unexpected_keys)
#             )
#     return model


# if __name__ == '__main__':
#     x = torch.randn([1, 3, 256, 128])
#     # model = build_osnet_backbone()
#     # x = torch.randn(16,64,64,32)
#     datamanager = torchreid.data.ImageDataManager(
#         root="/home/zyl/fast-reid/datasets",
#         sources="market1501",
#         targets="market1501",
#         height=256,
#         width=128,
#         batch_size_train=32,
#         batch_size_test=100,
#         transforms=["random_flip", "random_crop"]
#     )
#     model = torchreid.models.build_model(
#     name="shufflenet_v2_x1_0",
#     num_classes=datamanager.num_train_pids,
#     loss="softmax",
#     pretrained=False)
#     model.eval()
#     # lconv = LightConv3x3(64,64,nn.BatchNorm2d)
#     # flops, params = profile(model, inputs=(x,))
#     flops =FlopCountAnalysis(model,x)

#     # params = parameter_count(model)
#     # print(params)
#     # print(parameter_count_table(model))
#     # print(flops.by_module_and_operator())
#     print(fvcore.nn.flop_count_table(flops))

#     # print(flops.total() / 1e9, params.to / 1e6)  # flops单位G，para单位M
#     # stat(model, (3,256,128))


