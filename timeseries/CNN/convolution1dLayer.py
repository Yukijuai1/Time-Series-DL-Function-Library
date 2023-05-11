import torch.nn as nn


class Conv1dLayer(nn.Module):
    """
    一维卷积层的实现。

    Args:
        in_channels (int): 输入张量的通道数。
        out_channels (int): 输出张量的通道数。
        kernel_size (int): 卷积核的大小。
        stride (int): 卷积核的步幅。默认为1。
        padding (int): 输入张量的填充大小。默认为0。
        dilation (int): 卷积核的扩张率。默认为1。
        groups (int): 输入和输出之间连接的分组数。默认为1。
        bias (bool): 是否添加偏置项。默认为True。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
        super(Conv1dLayer, self).__init__()

        # 定义卷积层的核心参数
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # 前向传播过程
        x = self.conv1d(x)
        return x


def convolution1dLayer(in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return Conv1dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
