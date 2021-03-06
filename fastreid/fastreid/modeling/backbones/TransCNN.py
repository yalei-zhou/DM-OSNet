import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from thop import profile

# nn.SiLU()
# class SiLU(torch.nn.Module):  # export-friendly version of nn.SiLU()
#     @staticmethod
#     def forward(x):
#         return x * torch.sigmoid(x)
class SiLU(torch.nn.Module):
    """
    [https://arxiv.org/pdf/1710.05941.pdf]
    """
 
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
 
    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())

__all__ = ['TransCNN_Tiny', 'TransCNN_Small', 'TransCNN_Medium', 'TransCNN_Large']


class InvertedResidual(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3,
                 drop=0., act_layer=SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm_layer(hidden_dim),
            act_layer(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad, groups=hidden_dim, bias=False),
            norm_layer(hidden_dim),
            act_layer(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
            norm_layer(out_dim)
        )
        self.drop = nn.Dropout2d(drop, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        return x


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
            qkv = qkv.permute(1, 0, 2, 4, 6, 5, 7, 3)#permute???tensor??????????????????(3,64,1,8,4,8,8,64)
            qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size, self.head_dim)#(3,2048,64,64)
            q, k, v = qkv[0], qkv[1], qkv[2]#2048,64,64

            attn = (q @ k.transpose(-2, -1)) * self.scale#transpose????????????tensor???????????????.2048,64,64
            attn = attn.softmax(dim=-1)
            grid_x = (attn @ v).reshape(B, self.num_heads, grid_h, grid_w,#@????????????tensor?????????????????????
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


class Block(nn.Module):
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, expansion=4,
                 drop=0., drop_path=0., kernel_size=3, act_layer=SiLU,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()#DropPath/drop_path ???????????????????????????????????????????????????????????????????????????????????????????????????
        self.attn = Attention(dim, head_dim, grid_size=grid_size, ds_ratio=ds_ratio,
            drop=drop, norm_layer=norm_layer)
        self.conv = InvertedResidual(dim, hidden_dim=dim * expansion, out_dim=dim,
            kernel_size=kernel_size, drop=drop, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.conv(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, act_layer=SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.residual = nn.Conv2d(in_dim, out_dim, 1)
        self.norm1 = norm_layer(out_dim)
        self.norm2 = norm_layer(out_dim)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x1 = self.norm1(self.conv(x))
        x2 = self.norm2(self.residual(self.pool(x)))
        x = self.act(x1 + x2)
        return x


class TransCNN(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=1000, dims=[64, 128, 256, 512],
                 head_dim=64, expansions=[4, 4, 6, 6], grid_sizes=[8, 8, 8, 1],
                 ds_ratios=[8, 4, 2, 1], depths=[3, 4, 8, 3], drop_rate=0.,
                 drop_path_rate=0., act_layer=SiLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.depths = depths
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            norm_layer(16),
            act_layer(inplace=True),
            nn.Conv2d(16, dims[0], 3, padding=1, stride=2),
        )

        self.blocks = []
        kernel_sizes = [5, 3, 5, 3]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]#????????????1???????????????????????????start???end??????????????????step?????????
        for stage in range(len(dims)):
            self.blocks.append(nn.ModuleList([Block(
                dims[stage], head_dim, grid_size=grid_sizes[stage], ds_ratio=ds_ratios[stage],
                expansion=expansions[stage], drop=drop_rate, drop_path=dpr[sum(depths[:stage]) + i],
                kernel_size=kernel_sizes[stage], act_layer=act_layer,
                norm_layer=norm_layer) for i in range(depths[stage])]))
        self.blocks = nn.ModuleList(self.blocks)

        self.ds2 = Downsample(dims[0], dims[1], norm_layer=norm_layer)
        self.ds3 = Downsample(dims[1], dims[2], norm_layer=norm_layer)
        self.ds4 = Downsample(dims[2], dims[3], norm_layer=norm_layer)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(dims[-1], num_classes),
        )

        # init weights
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for stage in range(len(self.blocks)):
            for idx in range(self.depths[stage]):
                self.blocks[stage][idx].drop_path.drop_prob = dpr[cur + idx]
            cur += self.depths[stage]

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):#64,3,256,128
        x = self.patch_embed(x)
        for block in self.blocks[0]:#64,64,64,32
            x = block(x)
        x = self.ds2(x)
        for block in self.blocks[1]:
            x = block(x)
        x = self.ds3(x)
        for block in self.blocks[2]:
            x = block(x)
        x = self.ds4(x)
        for block in self.blocks[3]:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.classifier(x)

        return x


@register_model
def TransCNN_Tiny(pretrained=False, **kwargs):
    model = TransCNN(
        dims=[64, 128, 256, 512], head_dim=64, expansions=[4, 4, 4, 4],
        grid_sizes=[8, 8, 8, 1], ds_ratios=[8, 4, 2, 1], depths=[2, 2, 4, 2], **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def TransCNN_Small(pretrained=False, **kwargs):
    model = TransCNN(
        dims=[64, 128, 256, 512], head_dim=64, expansions=[4, 4, 6, 6],
        grid_sizes=[8, 8, 8, 1], ds_ratios=[8, 4, 2, 1], depths=[3, 4, 8, 3], **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def TransCNN_Medium(pretrained=False, **kwargs):
    model = TransCNN(
        dims=[64, 128, 320, 640], head_dim=64, expansions=[4, 4, 6, 6],
        grid_sizes=[8, 8, 8, 1], ds_ratios=[8, 4, 2, 1], depths=[3, 4, 12, 3], **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def TransCNN_Large(pretrained=False, **kwargs):
    model = TransCNN(
        dims=[64, 128, 384, 768], head_dim=64, expansions=[4, 4, 6, 6],
        grid_sizes=[8, 8, 8, 1], ds_ratios=[8, 4, 2, 1], depths=[3, 4, 16, 3], **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == "__main__":
    x = torch.randn(16,64,64,32)
    model = Attention(64, 16, grid_size=8, ds_ratio=8, drop=0., norm_layer=nn.BatchNorm2d)
    # out = model(x)
    # print(out)
    flops, params = profile(model, inputs=(x,))
    print(flops / 1e9, params / 1e6)  # flops??????G???para??????M
