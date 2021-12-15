import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch

from resnet import Backbone_ResNet152_in3

class DEM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DEM, self).__init__()
        self.global_max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1)


    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_max_pooling(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        h = x
        max_out, _ = torch.max(h, dim=1, keepdim=True)
        max_out = self.conv1(max_out)
        x = x * max_out.expand_as(x)
        return x


class ASPP(nn.Module):
    def __init__(self, outchannel):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=4, padding=4)
        self.conv0 = nn.Conv2d(outchannel, outchannel, kernel_size=1)

        self.conv = nn.Conv2d(5*outchannel, outchannel, kernel_size=1)

        self.rconv = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        out = torch.cat((x0, x1, x2, x3, x4), dim=1)
        out = self.conv(out)

        out = out + x
        out = self.rconv(out)


        return out


class olm(nn.Module):
    def __init__(self, outchannel, achannel):
        super(olm, self).__init__()
        self.conv1 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(outchannel, outchannel, kernel_size=3, dilation=4, padding=4)

        self.conv = nn.Conv2d(5*outchannel, outchannel, 3, padding=1)
        self.convs = nn.Sequential(
            nn.Conv2d(outchannel, achannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(achannel),
            nn.ReLU()
        )

        self.convf = nn.Conv2d(2*outchannel, outchannel, kernel_size=1)

        self.rconv = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

        self.rrconv = nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1)
        self.rrbn = nn.BatchNorm2d(outchannel)
        self.rrrelu = nn.ReLU()

        self.conv0 = nn.Conv2d(2*outchannel, outchannel, kernel_size=1)


    def  forward(self, x, ir):
        xx1 = x + ir
        xx1x = x * xx1
        xx1ir = ir * xx1
        xx = torch.cat((xx1x, xx1ir), dim=1)
        xx = self.conv0(xx)


        n = self.rrbn(self.rrconv(self.rconv(xx)))
        xx = self.rrrelu(xx + n)

        x1 = self.conv1(xx)
        x2 = self.conv2(xx)
        x3 = self.conv3(xx)
        x4 = self.conv4(xx)

        xp = torch.cat((xx, x1, x2, x3, x4), dim=1)
        xp = self.conv(xp)

        x_s = self.convs(xp)  #


        return x_s, xp


class EM(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(EM, self).__init__()
        self.conv = nn.Conv2d(2*inchannel, inchannel, kernel_size=1)

        self.rconv = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )
        self.rconv0 = nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1)
        self.rbn = nn.BatchNorm2d(inchannel)

        self.convfinal = nn.Conv2d(inchannel, outchannel, kernel_size=1)

    def forward(self, laster, current):

        out1 = torch.cat((laster, current), dim=1)
        out1 = self.conv(out1)

        x1 = laster * out1
        ir1 = current * out1
        f = x1 + ir1

        f = self.rbn(self.rconv0(self.rconv(f)))
        f = f + laster

        f = self.convfinal(f)

        return f


class EM2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(EM2, self).__init__()


    def forward(self, laster, current, high):


        f = laster + current + high

        return f


class seman(nn.Module):
    def __init__(self, inchannel):
        super(seman, self).__init__()
        self.rconv = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(inchannel, inchannel, kernel_size=1)
        self.convcat = nn.Conv2d(2*inchannel, inchannel, kernel_size=1)

        self.convedge = nn.Conv2d(1, 9, kernel_size=1)

    def forward(self, s1, s2, edge):
        s1 = torch.nn.functional.interpolate(s1, scale_factor=32, mode='bilinear')
        s2 = torch.nn.functional.interpolate(s2, scale_factor=16, mode='bilinear')
        s = torch.cat((s1, s2), dim=1)
        s = self.convcat(s)

        s = s + s1 + s2
        s = self.rconv(s)

        s = s * s1

        s = self.conv(s)

        edge = self.convedge(edge)


        se = s * edge
        s = se + s

        return s


class EGFNet(nn.Module):
    def __init__(self, n_classes):
        super(EGFNet, self).__init__()



        (
            self.layer1_rgb,
            self.layer2_rgb,
            self.layer3_rgb,
            self.layer4_rgb,
            self.layer5_rgb,
        ) = Backbone_ResNet152_in3(pretrained=True)



        self.em1 = olm(64, 2)
        self.em2 = olm(64, 2)
        self.em3 = olm(64, 2)
        self.em4 = olm(64, 9)
        self.em5 = olm(64, 9)



        self.resf4 = EM(64, 64)  # 512
        self.resf3 = EM2(64, 64)
        self.resf2 = EM2(64, 64)
        self.resf1 = EM2(64, 64)



        self.aspp = ASPP(64)
        self.finalconv = nn.Conv2d(64, 9, 1)
        self.convb3 = nn.Conv2d(2, 2, 1)

        self.rgbconv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.rgbconv2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.rgbconv3 = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.rgbconv4 = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.rgbconv5 = nn.Conv2d(2048, 64, kernel_size=3, padding=1)

        self.convedge = nn.Conv2d(1, 9, kernel_size=1)




        self.semantic = seman(9)



    def forward(self, rgb, depth, label):
        x = rgb
        ir = depth[:, :1, ...]
        ir = torch.cat((ir, ir, ir), dim=1)
        edge = label
        edge = torch.unsqueeze(edge, 1).float()






        x1 = self.layer1_rgb(x)
        ir1 = self.layer1_rgb(ir)



        x2 = self.layer2_rgb(x1)
        ir2 = self.layer2_rgb(ir1)


        x3 = self.layer3_rgb(x2)
        ir3 = self.layer3_rgb(ir2)


        x4 = self.layer4_rgb(x3)
        ir4 = self.layer4_rgb(ir3)


        x5 = self.layer5_rgb(x4)
        ir5 = self.layer5_rgb(ir4)


        x1 = self.rgbconv1(x1)
        x2 = self.rgbconv2(x2)
        x3 = self.rgbconv3(x3)
        x4 = self.rgbconv4(x4)
        x5 = self.rgbconv5(x5)

        ir1 = self.rgbconv1(ir1)
        ir2 = self.rgbconv2(ir2)
        ir3 = self.rgbconv3(ir3)
        ir4 = self.rgbconv4(ir4)
        ir5 = self.rgbconv5(ir5)



        s5, out5 = self.em5(x5, ir5)
        s4, out4 = self.em4(x4, ir4)
        b3, out3 = self.em3(x3, ir3)
        b2, out2 = self.em2(x2, ir2)
        b1, out1 = self.em1(x1, ir1)

        S = self.semantic(s5, s4, edge)
        b3 = torch.nn.functional.interpolate(b3, scale_factor=8, mode='bilinear')
        b3 = self.convb3(b3)
        b3 = b3 * edge

        b2 = torch.nn.functional.interpolate(b2, scale_factor=4, mode='bilinear')
        b2 = self.convb3(b2)
        b2 = b2 * edge

        b1 = torch.nn.functional.interpolate(b1, scale_factor=2, mode='bilinear')
        b1 = self.convb3(b1)
        b1 = b1 * edge


        out51 = self.aspp(out5)
        out51 = torch.nn.functional.interpolate(out51, scale_factor=2, mode='bilinear')

        out41 = self.resf4(out51, out4)
        f4 = torch.nn.functional.interpolate(out41, scale_factor=2, mode='bilinear')

        high3 = torch.nn.functional.interpolate(out51, scale_factor=2, mode='bilinear')
        out31 = self.resf3(f4, out3, high3)
        f3 = torch.nn.functional.interpolate(out31, scale_factor=2, mode='bilinear')


        high2 = torch.nn.functional.interpolate(out51, scale_factor=4, mode='bilinear')
        out21 = self.resf2(f3, out2, high2)
        f2 = torch.nn.functional.interpolate(out21, scale_factor=2, mode='bilinear')

        high1 = torch.nn.functional.interpolate(out51, scale_factor=8, mode='bilinear')
        out11 = self.resf1(f2, out1, high1)
        out11 = torch.nn.functional.interpolate(out11, scale_factor=2, mode='bilinear')

        semantic_out = self.finalconv(out11)
        edge1 = self.convedge(edge)
        semantic_outee = semantic_out * edge1
        semantic_out = semantic_outee + semantic_out


        return semantic_out, S, b3, b2, b1

if __name__ == '__main__':
    EGFNet(9)
