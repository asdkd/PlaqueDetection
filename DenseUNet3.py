import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#Dense UNet
class DenseUNet(nn.Module):  
    def __init__(self, subSlices, categories):
        super(DenseUNet,self).__init__()
        # chann 64->128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=subSlices,out_channels=64,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.enc_layer_1 = nn.Sequential(
            self.enc_conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            # nn.ReLU(),
            nn.Dropout(p=0)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # chann 128->256
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.enc_layer_2 = nn.Sequential(
            self.enc_conv2,
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            # nn.ReLU(),
            nn.Dropout(p=0)
        )
        # chann 256->512
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.enc_layer_3 = nn.Sequential(
            self.enc_conv3,
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            # nn.ReLU(),
            nn.Dropout(p=0)
        )
        # chann 512->1024
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.enc_layer_4 = nn.Sequential(
            self.enc_conv4,
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            # nn.ReLU(),
            nn.Dropout(p=0)
        )
        self.drop = nn.Dropout(p=0.5)
        # chann 1024->1024+1024
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.enc_layer_5 = nn.Sequential(
            self.enc_conv5,
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1),
            # nn.ReLU(),
            nn.Dropout(p=0)
        )

        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=512,kernel_size=2,stride=1,padding=1),
            # nn.ReLU()
        )
        self.dec_conv4_ = nn.Sequential(
            nn.Conv2d(in_channels=1536,out_channels=512,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.dec_layer_4 = nn.Sequential(
            self.dec_conv4_,
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            # nn.ReLU(),
            nn.Dropout(p=0)
        )

        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256,kernel_size=2,stride=1,padding=1),
            # nn.ReLU()
        )
        self.dec_conv3_ = nn.Sequential(
            nn.Conv2d(in_channels=768,out_channels=256,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.dec_layer_3 = nn.Sequential(
            self.dec_conv3_,
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            # nn.ReLU(),
            nn.Dropout(p=0)
        )

        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128,kernel_size=2,stride=1,padding=1),
            # nn.ReLU()
        )
        self.dec_conv2_ = nn.Sequential(
            nn.Conv2d(in_channels=384,out_channels=128,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.dec_layer_2 = nn.Sequential(
            self.dec_conv2_,
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            # nn.ReLU(),
            nn.Dropout(p=0)
        )

        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64,kernel_size=2,stride=1,padding=1),
            # nn.ReLU()
        )
        self.dec_conv1_ = nn.Sequential(
            nn.Conv2d(in_channels=192,out_channels=64,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.dec_layer_1 = nn.Sequential(
            self.dec_conv1_,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            # nn.ReLU(),
            nn.Dropout(p=0)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=categories,kernel_size=3,stride=1,padding=1),
            # nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=categories, out_channels=categories,kernel_size=1,stride=1,padding=0),
            nn.Sigmoid()
        )

    def forward(self,x,mask):
        conv1_1 = self.enc_conv1(x)
        drop1_2 = self.enc_layer_1(x)
        Merge1 = torch.cat([conv1_1,drop1_2],dim=1)
        pool1 = self.maxpool(Merge1)

        conv2_1 = self.enc_conv2(pool1)
        drop2_2 = self.enc_layer_2(pool1)
        Merge2 = torch.cat([conv2_1,drop2_2],dim=1)
        pool2 = self.maxpool(Merge2)

        conv3_1 = self.enc_conv3(pool2)
        drop3_2 = self.enc_layer_3(pool2)
        Merge3 = torch.cat([conv3_1,drop3_2],dim=1)
        pool3 = self.maxpool(Merge3)

        conv4_1 = self.enc_conv4(pool3)
        drop4_2 = self.enc_layer_4(pool3)
        Merge4 = torch.cat([conv4_1,drop4_2],dim=1)
        drop4 = self.drop(Merge4)
        pool4 = self.maxpool(drop4)

        conv5_1 = self.enc_conv5(pool4)
        drop5_2 = self.enc_layer_5(pool4)
        Merge5 = torch.cat([conv5_1,drop5_2],dim=1)
        drop5 = self.drop(Merge5)

        up6 = self.dec_conv4(drop5)
        up6 = F.interpolate(up6,scale_factor=2)
        up6 = up6[:,:,1:-1,1:-1]
        merge6 = torch.cat([drop4,up6],dim=1)
        conv6_1 = self.dec_conv4_(merge6)
        drop6_2 = self.dec_layer_4(merge6)
        Merge6 = torch.cat([conv6_1,drop6_2],dim=1)

        up7 = self.dec_conv3(Merge6)
        up7 = F.interpolate(up7,scale_factor=2)
        up7 = up7[:,:,1:-1,1:-1]
        merge7 = torch.cat([Merge3,up7],dim=1)
        conv7_1 = self.dec_conv3_(merge7)
        drop7_2 = self.dec_layer_3(merge7)
        Merge7 = torch.cat([conv7_1,drop7_2],dim=1)

        up8 = self.dec_conv2(Merge7)
        up8 = F.interpolate(up8,scale_factor=2)
        up8 = up8[:,:,1:-1,1:-1]
        merge8 = torch.cat([Merge2,up8],dim=1)
        conv8_1 = self.dec_conv2_(merge8)
        drop8_2 = self.dec_layer_2(merge8)
        Merge8 = torch.cat([conv8_1,drop8_2],dim=1)

        up9 = self.dec_conv1(Merge8)
        up9 = F.interpolate(up9,scale_factor=2)
        up9 = up9[:,:,1:-1,1:-1]
        merge9 = torch.cat([Merge1,up9],dim=1)
        conv9_1 = self.dec_conv1_(merge9)
        drop9_2 = self.dec_layer_1(merge9)
        Merge9 = torch.cat([conv9_1,drop9_2],dim=1)

        conv9 = self.conv(Merge9)*mask
        conv10 = self.conv2(conv9)*mask + 1e-10

        return conv10

        


