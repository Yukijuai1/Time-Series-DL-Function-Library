import torch
import torch.nn as nn


class Conv2D(nn.Module):
    """
    二维卷积层
    Args:
        in_channels (int): 输入张量的通道数
        out_channels (int): 输出张量的通道数
        kernel_size (int or tuple): 卷积核大小
        stride (int or tuple): 卷积步长，默认为1
        padding (int or tuple): 卷积填充，默认为0
        dilation (int or tuple): 卷积扩张率，默认为1
        groups (int): 分组卷积的组数，默认为1
        bias (bool): 是否使用偏置，默认为True
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super(Conv2D, self).__init__()
        # 定义卷积层的权重
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size))
        # 是否使用偏置
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        # 卷积参数
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        # 卷积计算
        out = nn.functional.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                                   dilation=self.dilation, groups=self.groups)
        return out
    
def convolution2dLayer(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return Conv2D(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
