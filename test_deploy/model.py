import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv2D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv3D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#-------------------------------------------

class Down2D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Down2D,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_builder(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder):
        super(Down3D,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            conv_builder(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


#-------------------------------------------


class Up2D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder, bilinear=True):
        super(Up2D,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = conv_builder(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv_builder(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder, bilinear=True):
        super(Up3D,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = conv_builder(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv_builder(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffD // 2, diffD - diffD // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffW // 2, diffW - diffW // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#-------------------------------------------

class Tail2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tail2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Tail3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tail3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#-------------------------------------------

class UNet(nn.Module):
    def __init__(self, stem, down, up, tail, width, conv_builder, n_channels=1, n_classes=2, bilinear=True, dropout_flag=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.width = width
        self.dropout_flag = dropout_flag
        factor = 2 if bilinear else 1

        self.inc = stem(n_channels, width[0])
        self.down1 = down(width[0], width[1], conv_builder)
        self.down2 = down(width[1], width[2], conv_builder)
        self.down3 = down(width[2], width[3], conv_builder)
        self.down4 = down(width[3], width[4] // factor, conv_builder)
        self.up1 = up(width[4], width[3] // factor, conv_builder, bilinear=self.bilinear)
        self.up2 = up(width[3], width[2]// factor, conv_builder, bilinear=self.bilinear)
        self.up3 = up(width[2], width[1] // factor, conv_builder, bilinear=self.bilinear)
        self.up4 = up(width[1], width[0], conv_builder, bilinear=self.bilinear)
        self.dropout = nn.Dropout(p=0.5)
        self.outc = tail(width[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.dropout_flag:
            x = self.dropout(x)
        logits = self.outc(x)
        return logits



def unet_2d(**kwargs):
    return UNet(stem=DoubleConv2D,
                down=Down2D,
                up=Up2D,
                tail=Tail2D,
                width=[64,128,256,512,1024],
                conv_builder=DoubleConv2D,
                **kwargs)


def unet_3d(**kwargs):
    return UNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                conv_builder=DoubleConv3D,
                **kwargs)


if __name__ == "__main__":
  
  net = unet_2d(n_channels=1, n_classes=2, bilinear=True)
#   net = unet_3d(n_channels=1, n_classes=2, bilinear=True)

  from torchsummary import summary
  import os 
  os.environ['CUDA_VISIBLE_DEVICES'] = '6'

  summary(net.cuda(),input_size=(1,128,128),batch_size=1,device='cuda')

