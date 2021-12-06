import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun

class Single_level_densenet(nn.Module):
    def __init__(self, filters, num_conv=4):
        super(Single_level_densenet, self).__init__()
        self.num_conv = num_conv
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(self.num_conv):
            self.conv_list.append(nn.Conv2d(filters, filters, 3, padding=1).apply(weights_init('kaiming')))
            self.bn_list.append(nn.BatchNorm2d(filters))

    def forward(self, x):
        outs = []
        outs.append(x)
        for i in range(self.num_conv):
            temp_out = self.conv_list[i](outs[i])
            if i > 0:
                for j in range(i):
                    temp_out += outs[j]
            outs.append(F.relu(self.bn_list[i](temp_out)))
        out_final = outs[-1]
        del outs
        return out_final


class Down_sample(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(Down_sample, self).__init__()
        self.down_sample_layer = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        y = self.down_sample_layer(x)
        return y, x


class Upsample_n_Concat(nn.Module):
    def __init__(self, filters):
        super(Upsample_n_Concat, self).__init__()
        self.upsample_layer = nn.ConvTranspose2d(filters, filters, 4, padding=1, stride=2)
        self.conv = nn.Conv2d(2 * filters, filters, 3, padding=1).apply(weights_init('kaiming'))
        self.bn = nn.BatchNorm2d(filters)

    def forward(self, x, y):
        x = self.upsample_layer(x)
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.bn(self.conv(x)))
        return x


class Dense_Unet(nn.Module):
    def __init__(self, subSlices, categories):
        self.subSlices = subSlices
        self.categories = categories
        num_conv = 2
        filters = 64
        in_chan = subSlices
        out_chan = categories
        super(Dense_Unet, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, filters, 1)
        self.d1 = Single_level_densenet(filters, num_conv)
        self.down1 = Down_sample()
        self.d2 = Single_level_densenet(filters, num_conv)
        self.down2 = Down_sample()
        self.d3 = Single_level_densenet(filters, num_conv)
        self.down3 = Down_sample()
        self.d4 = Single_level_densenet(filters, num_conv)
        self.down4 = Down_sample()
        self.bottom = Single_level_densenet(filters, num_conv)
        self.up4 = Upsample_n_Concat(filters)
        self.u4 = Single_level_densenet(filters, num_conv)
        self.up3 = Upsample_n_Concat(filters)
        self.u3 = Single_level_densenet(filters, num_conv)
        self.up2 = Upsample_n_Concat(filters)
        self.u2 = Single_level_densenet(filters, num_conv)
        self.up1 = Upsample_n_Concat(filters)
        self.u1 = Single_level_densenet(filters, num_conv)
        self.outconv = nn.Conv2d(filters, out_chan, 3, padding=1).apply(weights_init('kaiming'))
        self.relu = nn.ReLU()
        self.outconv2 = nn.Conv2d(out_chan, out_chan, 1)
        self.sigmoid = nn.Sigmoid()

    #         self.outconvp1 = nn.Conv2d(filters,out_chan, 1)
    #         self.outconvm1 = nn.Conv2d(filters,out_chan, 1)

    def forward(self, x, mask):
        #bsz = x.shape[0]
        x = self.conv1(x)
        x, y1 = self.down1(self.d1(x))
        x, y2 = self.down1(self.d2(x))
        x, y3 = self.down1(self.d3(x))
        x, y4 = self.down1(self.d4(x))
        x = self.bottom(x)
        x = self.u4(self.up4(x, y4))
        x = self.u3(self.up3(x, y3))
        x = self.u2(self.up2(x, y2))
        x = self.u1(self.up1(x, y1))
        x1 = self.outconv(x)
        x1 = self.relu(x1)
        x1 = self.outconv2(x1)
        x1 = self.sigmoid(x1)
        x2 = x1*mask 
        #         xm1 = self.outconvm1(x)
        #         xp1 = self.outconvp1(x)

        return x2#x1