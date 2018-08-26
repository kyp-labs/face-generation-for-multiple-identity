"""Model based on the paper.

U-Net: Convolutional Networks for Biomedical Image Segmentation
<https://arxiv.org/abs/1505.04597>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import kaiming_normal_


def upsample(x, factor):
    """Upsample tensor.

    Args:
        factor (int): factor to upsample.

    Return: upsampled tensor.
    """
    return F.upsample(x, scale_factor=factor)


class UConv2d(nn.Module):
    """Simple Convolutional Network class for U-Net."""

    def __init__(self, in_channels, out_channels, nonlinearity,
                 kernel_size=3, padding=1):
        """constructor.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            nonlinearity: nonlinearity function
            kernel_size (int): Filter kernel size, Default is 3.
            padding: padding size, Default is 1.
        """
        super(UConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding)
        kaiming_normal_(self.conv.weight)
        self.nonlinearity = nonlinearity
        self.out_channels = out_channels

    def forward(self, x):
        """forward.

        Args:
            x (tensor): [batch_size, in_channels, height, width],
                        input tensor.

        Returns:
            x (tensor): [batch_size, out_channels, height', width'],
                        output tensor.

        """
        x = self.conv(x)
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)
        x = nn.BatchNorm2d(self.out_channels)(x)
        return x


class In(nn.Module):
    """U-Net's first network class."""

    def __init__(self, in_channels, out_channels):
        """constructor.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super(In, self).__init__()
        self.conv1 = UConv2d(in_channels, out_channels, nn.ReLU())
        self.conv2 = UConv2d(out_channels, out_channels, nn.ReLU())

    def forward(self, x):
        """forward.

        Args:
            x (tensor): [batch_size, in_channels, height, width]

        Returns:
            x (tensor): [batch_size, out_channels, height, width]

        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Down(nn.Module):
    """U-net's downsampling network class."""

    def __init__(self, in_channels, out_channels):
        """constructor.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = UConv2d(in_channels, out_channels, nn.ReLU())
        self.conv2 = UConv2d(out_channels, out_channels, nn.ReLU())

    def forward(self, x):
        """forward.

        Args:
            x (tensor): [batch_size, in_channels, height, width]

        Returns:
            x (tensor): [batch_size, out_channels, height//2, width//2]

        """
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up(nn.Module):
    """U-net's upsampling network class."""

    def __init__(self, in_channels, out_channels):
        """constructor.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
        """
        super(Up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        self.conv1 = UConv2d(in_channels, out_channels, nn.ReLU())
        self.conv2 = UConv2d(out_channels, out_channels, nn.ReLU())

    def forward(self, x1, x2):
        """forward.

        Args:
            x1 (tensor): [batch_size, in_channels, height, width]
            x2 (tensor): [batch_size, in_channels, height, width]

        Returns:
            x (tensor): [batch_size, out_channels, 2*height, 2*width]

        """
        x1 = upsample(x1, 2)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffY // 2, int(diffY / 2),
                        diffX // 2, int(diffX / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Out(nn.Module):
    """U-net's last network class."""

    def __init__(self, in_channels, num_classes):
        """constructor.

        Args:
            in_channels (int): The number of input channels.
            num_classes (int): The number of output classes.
        """
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        """forward.

        Args:
            x (tensor): [batch_size, in_channels, height, width]

        Returns:
            x (tensor): [batch_size, num_classes]

        """
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """U-Net model class.

    Based on "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    <https://arxiv.org/abs/1505.04597.pdf>
    """

    def __init__(self, num_channels, num_classes):
        """constructor.

        Args:
            num_channels (int): The number of input image channels.
            num_classes (int): The number of output classes.
        """
        super(UNet, self).__init__()
        self.inconv = In(num_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outconv = Out(64, num_classes)

    def forward(self, x):
        """forward.

        Args:
            x (tensor): [batch_size, num_channels, height, width]

        Returns:
            x (tensor): [batch_size, num_classes]

        """
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outconv(x)
        return x
