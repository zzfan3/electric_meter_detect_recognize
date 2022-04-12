# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownDoubleSepConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownDoubleSepConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*4, 1),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*4, in_ch*4, 3, padding=1, groups=in_ch*4),
            nn.BatchNorm2d(in_ch*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch*4, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpDoubleSepConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpDoubleSepConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.size()[2] < x2.size()[2]:
            x1 = F.pad(x1, (0, 0, 0, x2.size()[2] - x1.size()[2]))
        if x1.size()[3] < x2.size()[3]:
            x1 = F.pad(x1, (0, x2.size()[3] - x1.size()[3], 0, 0))
        if x1.size()[2] > x2.size()[2]:
            x2 = F.pad(x1, (0, 0, 0, x1.size()[2] - x2.size()[2]))
        if x1.size()[3] > x2.size()[3]:
            x2 = F.pad(x1, (0, x1.size()[3] - x2.size()[3], 0, 0))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class SepDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SepDown, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            DownDoubleSepConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SepUp(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(SepUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = UpDoubleSepConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.size()[2] < x2.size()[2]:
            x1 = F.pad(x1, (0, 0, 0, x2.size()[2] - x1.size()[2]))
        if x1.size()[3] < x2.size()[3]:
            x1 = F.pad(x1, (0, x2.size()[3] - x1.size()[3], 0, 0))
        if x1.size()[2] > x2.size()[2]:
            x2 = F.pad(x1, (0, 0, 0, x1.size()[2] - x2.size()[2]))
        if x1.size()[3] > x2.size()[3]:
            x2 = F.pad(x1, (0, x1.size()[3] - x2.size()[3], 0, 0))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet3x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3x, self).__init__()
        self.inc = InConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up4 = Up(768, 256)
        self.up3 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(96, 32)

        self.out1 = OutConv(32, out_channels)
        self.out2 = OutConv(64, out_channels)
        self.out3 = OutConv(128, out_channels)
        self.out4 = OutConv(256, out_channels)
        self.out5 = OutConv(512, out_channels)

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)

        up4 = self.up4(f5, f4)
        s4 = self.out4(up4)
        up3 = self.up3(up4, f3)
        s3 = self.out3(up3)
        up2 = self.up2(up3, f2)
        s2 = self.out2(up2)
        up1 = self.up1(up2, f1)
        s1 = self.out1(up1)
        return s1, s2, s3, s4


class UNet2x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2x, self).__init__()
        self.inc = InConv(in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        self.up4 = Up(384, 128)
        self.up3 = Up(192, 64)
        self.up2 = Up(96, 32)
        self.up1 = Up(48, 16)

        self.out1 = OutConv(16, out_channels)
        self.out2 = OutConv(32, out_channels)
        self.out3 = OutConv(64, out_channels)
        self.out4 = OutConv(128, out_channels)
        self.out5 = OutConv(256, out_channels)

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)

        up4 = self.up4(f5, f4)
        s4 = self.out4(up4)
        up3 = self.up3(up4, f3)
        s3 = self.out3(up3)
        up2 = self.up2(up3, f2)
        s2 = self.out2(up2)
        up1 = self.up1(up2, f1)
        s1 = self.out1(up1)
        return s1, s2, s3, s4


class UNet1x(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1x, self).__init__()
        self.inc = InConv(in_channels, 16)
        self.down1 = SepDown(16, 32)
        self.down2 = SepDown(32, 64)
        self.down3 = SepDown(64, 128)
        self.down4 = SepDown(128, 256)

        self.up4 = SepUp(384, 128)
        self.up3 = SepUp(192, 64)
        self.up2 = SepUp(96, 32)
        self.up1 = SepUp(48, 16)

        self.out1 = OutConv(16, out_channels)
        self.out2 = OutConv(32, out_channels)
        self.out3 = OutConv(64, out_channels)
        self.out4 = OutConv(128, out_channels)
        self.out5 = OutConv(256, out_channels)

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)

        up4 = self.up4(f5, f4)
        s4 = self.out4(up4)
        up3 = self.up3(up4, f3)
        s3 = self.out3(up3)
        up2 = self.up2(up3, f2)
        s2 = self.out2(up2)
        up1 = self.up1(up2, f1)
        s1 = self.out1(up1)
        return s1, s2, s3, s4
