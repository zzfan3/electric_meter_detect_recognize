import torch.nn as nn
import torch
import torch.nn.functional as F


class OUNet5(nn.Module):
    """
    Base UNet For Five Level

    Return:
        predicted segment
        s1
    """
    def __init__(self, in_channels, out_channels):
        super(OUNet5, self).__init__()
        self.inc = inconv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up4 = Up(768, 256)
        self.up3 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(96, 32)

        self.out1 = outconv(32, out_channels)
        self.out2 = outconv(64, out_channels)
        self.out3 = outconv(128, out_channels)
        self.out4 = outconv(256, out_channels)
        self.out5 = outconv(512, out_channels)

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)

        s5 = torch.sigmoid(self.out5(f5))
        up4 = self.up4(f5, f4)
        s4 = torch.sigmoid(self.out4(up4))
        up3 = self.up3(up4, f3)
        s3 = torch.sigmoid(self.out3(up3))
        up2 = self.up2(up3, f2)
        s2 = torch.sigmoid(self.out2(up2))
        up1 = self.up1(up2, f1)
        s1 = torch.sigmoid(self.out1(up1))
        return s1, s2, s3, s4


class DUNet5(nn.Module):
    """
    Double output UNet For No Five Level

    Return:
        predicted segment
        s1
    """
    def __init__(self, in_channels, out_channels):
        super(DUNet5, self).__init__()
        self.inc = inconv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up4_A = Up(768, 256)
        self.up3_A = Up(384, 128)
        self.up2_A = Up(192, 64)
        self.up1_A = Up(96, 32)

        self.up4_B = Up(768, 256)
        self.up3_B = Up(384, 128)
        self.up2_B = Up(192, 64)
        self.up1_B = Up(96, 32)

        self.out1_A = outconv(32, out_channels)
        self.out1_B = outconv(32, out_channels)

    def forward(self, x):
        f1 = self.inc(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)

        # for branch A
        # s5 = torch.sigmoid(self.out5(f5))
        out_up4_A = self.up4_A(f5, f4)
        # s4 = torch.sigmoid(self.out4(up4))
        out_up3_A = self.up3_A(out_up4_A, f3)
        # s3 = torch.sigmoid(self.out3(up3))
        out_up2_A = self.up2_A(out_up3_A, f2)
        # s2 = torch.sigmoid(self.out2(up2))
        out_up1_A = self.up1_A(out_up2_A, f1)
        out_s1_A = torch.sigmoid(self.out1_A(out_up1_A))

        # for branch B
        # s5 = torch.sigmoid(self.out5(f5))
        out_up4_B = self.up4_B(f5, f4)
        # s4 = torch.sigmoid(self.out4(up4))
        out_up3_B = self.up3_B(out_up4_B, f3)
        # s3 = torch.sigmoid(self.out3(up3))
        out_up2_B = self.up2_B(out_up3_B, f2)
        # s2 = torch.sigmoid(self.out2(up2))
        out_up1_B = self.up1_B(out_up2_B, f1)
        out_s1_B = torch.sigmoid(self.out1_B(out_up1_B))
        return out_s1_A, out_s1_B


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

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


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class double_conv(nn.Module):
    '''
    (conv => BN => ReLU) * 2
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
