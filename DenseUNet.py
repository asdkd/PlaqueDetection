from typing_extensions import Concatenate
import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

#Dense UNet
class DenseUNet(nn.Module):  
    def __init__(self, subSlices, categories):

        super(DenseUNet,self).__init__()
        # self.first_conv = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=subSlices,
        #         out_channels=64,
        #         kernel_size=(1, 1),
        #         stride=1,

        #     ).apply(weights_init('kaiming')),
        #     nn.BatchNorm2d(64),
        #   #   nn.LeakyReLU(0.2)
        #   nn.ReLU()
        # )
        #Dense block & DownSampling
        ###dense1 64 -> 128### 
        self.dense1_c1  = nn.Conv2d(in_channels=7,out_channels=64,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        self.d1_bn1     = nn.BatchNorm2d(64,affine=True)
        self.dense1_c2  = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d1_bn2     = nn.BatchNorm2d(64)
      #   self.dense1_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
      #   self.d1_bn3     = nn.BatchNorm2d(32)

        #down sampling
        # self.DS1 = nn.Sequential(
        #      nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(2,2),stride=(2,2)).apply(weights_init('kaiming')),
        #      nn.BatchNorm2d(128),
        #   #    nn.LeakyReLU(0.2),
        #      nn.ReLU()
        # )
        
      
        ###dense2 32 -> 32
        self.dense2_c1  = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        self.d2_bn1     = nn.BatchNorm2d(128,affine=True)
        self.dense2_c2  = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d2_bn2     = nn.BatchNorm2d(128)
      #   self.dense2_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
      #   self.d2_bn3     = nn.BatchNorm2d(32)

        #down sampling
        # self.DS2 = nn.Sequential(
        #      nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(2,2),stride=(2,2)).apply(weights_init('kaiming')),
        #      nn.BatchNorm2d(256),
        #   #    nn.LeakyReLU(0.2),
        #   nn.ReLU()
        # )

        ###dense3 32 -> 32
        self.dense3_c1  = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        self.d3_bn1     = nn.BatchNorm2d(256,affine=True)
        self.dense3_c2  = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d3_bn2     = nn.BatchNorm2d(256)
      #   self.dense3_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
      #   self.d3_bn3     = nn.BatchNorm2d(32)

        #down sampling
        # self.DS3 = nn.Sequential(
        #      nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(2,2),stride=(2,2)).apply(weights_init('kaiming')),
        #      nn.BatchNorm2d(512),
        #   #    nn.LeakyReLU(0.2),
        #   nn.ReLU()
        # ) 
        
        ###dense4 32 -> 32
        self.dense4_c1  = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))#concat
        self.d4_bn1     = nn.BatchNorm2d(512,affine=True)
        self.dense4_c2  = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d4_bn2     = nn.BatchNorm2d(512)
      #   self.dense4_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
      #   self.d4_bn3     = nn.BatchNorm2d(32)

        #down sampling
        # self.DS4 = nn.Sequential(
        #      nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=(2,2),stride=(2,2)).apply(weights_init('kaiming')),
        #      nn.BatchNorm2d(1024),
        #   #    nn.LeakyReLU(0.2),
        #   nn.ReLU()
        # )

        ###dense5 32 -> 32
        self.dense5_c1  = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))#concat
        self.d5_bn1     = nn.BatchNorm2d(1024,affine=True)
        self.dense5_c2  = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d5_bn2     = nn.BatchNorm2d(1024)
      #   self.dense5_c3  = nn.Conv2d(in_channels=96,out_channels=32,kernel_size=(3,3),stride=1,padding=(1,1))
      #   self.d5_bn3     = nn.BatchNorm2d(32)
        

        ###dense6 32 -> 32
        self.dense6_c1  = nn.Conv2d(in_channels=2048,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))#concat
        self.d6_bn1     = nn.BatchNorm2d(512,affine=True)
      #   up sampling
        # self.UP1 = nn.Sequential(
        #      nn.ConvTranspose2d(in_channels=512,out_channels=512,kernel_size=(2,2),stride=(2,2)),
        #      nn.BatchNorm2d(512),
        #   #    nn.LeakyReLU(0.2)
        #   nn.ReLU()
        # ) 
        self.dense6_c2  = nn.Conv2d(in_channels=1024+512,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d6_bn2     = nn.BatchNorm2d(512)
        self.dense6_c3  = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d6_bn3     = nn.BatchNorm2d(512)

        ###dense7 32 -> 32
        self.dense7_c1  = nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))#concat
        self.d7_bn1     = nn.BatchNorm2d(256,affine=True)
        #up sampling
        # self.UP2 = nn.Sequential(
        #      nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=(2,2),stride=(2,2)).apply(weights_init('kaiming')),
        #      nn.BatchNorm2d(256),
        #   #    nn.LeakyReLU(0.2)
        #   nn.ReLU()
        # ) 
        self.dense7_c2  = nn.Conv2d(in_channels=512+256,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d7_bn2     = nn.BatchNorm2d(256)
        self.dense7_c3  = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d7_bn3     = nn.BatchNorm2d(256)

        ###dense8 32 -> 32
        self.dense8_c1  = nn.Conv2d(in_channels=512,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))#concat
        self.d8_bn1     = nn.BatchNorm2d(128,affine=True)
        #up sampling
        # self.UP3 = nn.Sequential(
        #      nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=(2,2),stride=(2,2)).apply(weights_init('kaiming')),
        #      nn.BatchNorm2d(128),
        #   #    nn.LeakyReLU(0.2)
        #   nn.ReLU()
        # ) 
        self.dense8_c2  = nn.Conv2d(in_channels=256+128,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d8_bn2     = nn.BatchNorm2d(128)
        self.dense8_c3  = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d8_bn3     = nn.BatchNorm2d(128)



        ###dense9 32 -> 32
        self.dense9_c1  = nn.Conv2d(in_channels=256,out_channels=64,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))#concat
        self.d9_bn1     = nn.BatchNorm2d(64,affine=True)
        # self.UP4 = nn.Sequential(
        #      nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=(2,2),stride=(2,2)).apply(weights_init('kaiming')),
        #      nn.BatchNorm2d(64),
        #   #    nn.LeakyReLU(0.2)
        #   nn.ReLU()
        # ) 
        self.dense9_c2  = nn.Conv2d(in_channels=128+64,out_channels=64,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d9_bn2     = nn.BatchNorm2d(64)
        self.dense9_c3  = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming'))
        # self.d9_bn3     = nn.BatchNorm2d(64)


        self.last_conv = nn.Sequential(

            nn.Conv2d(in_channels=128,out_channels=categories,kernel_size=(3,3),stride=1,padding=(1,1)).apply(weights_init('kaiming')),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=categories,
                out_channels=categories,
                kernel_size=(1, 1),
                stride=1,
               ),
            nn.Sigmoid()
        )

        # self.FC = nn.Linear(in_features=128, out_features=categories)
        
        


    def forward(self,x,mask):
        #活性化関数(activation function)の定義
        conv_activation = torch.nn.LeakyReLU(0.2)
        # conv_activation = torch.nn.ReLU()
        #print("x: ",x.shape)
        # first  = self.first_conv(x)
        first = x
        #print("first: ",first.shape)
      #   dense1_1 = conv_activation( self.d1_bn1(self.dense1_c1(first)))  # 16*64*128*128
      #   dense1_2 = conv_activation( self.d1_bn2(self.dense1_c2(dense1_1) )) # 16*64*128*128
        running_mean = self.d1_bn1.running_mean
        running_var = self.d1_bn1.running_var
        dense1_1 = conv_activation( self.d1_bn1( conv_activation(self.dense1_c1(first)) ))  
        dense1_2 = conv_activation( self.dense1_c2(dense1_1) ) 
        Concatenate1 = torch.cat((dense1_1,dense1_2),dim=1) # 16*128*128*128
      #   D1       = self.DS1(Concatenate1) # 16*128*64*64
        D1 = nn.MaxPool2d(kernel_size=(2,2))(Concatenate1)
      #   D1       = self.DS_maxpool(dense1_3)
     
        #print("D1: ",D1.shape)
      #   dense2_1 = conv_activation( self.d2_bn1(self.dense2_c1(D1)))  # 16*128*64*64
      #   dense2_2 = conv_activation( self.d2_bn2(self.dense2_c2(dense2_1 ) )) # 16*128*64*64
        dense2_1 = conv_activation( self.d2_bn1( conv_activation(self.dense2_c1(D1)) ))  
        dense2_2 = conv_activation( self.dense2_c2(dense2_1) ) 
        Concatenate2 = torch.cat((dense2_1,dense2_2),dim=1) # 16*256*64*64
      #   D2       = self.DS2(Concatenate2) # 16*256*32*32
        D2 = nn.MaxPool2d(kernel_size=(2,2))(Concatenate2)
      #   D2       = self.DS_maxpool(dense2_3)
     
        #print("D2: ",D2.shape)
      #   dense3_1 = conv_activation( self.d3_bn1(self.dense3_c1(D2))) # 16*256*32*32
      #   dense3_2 = conv_activation( self.d3_bn2(self.dense3_c2(dense3_1 ) )) # 16*256*32*32
        dense3_1 = conv_activation( self.d3_bn1( conv_activation(self.dense3_c1(D2)) ))  
        dense3_2 = conv_activation( self.dense3_c2(dense3_1) ) 
        Concatenate3 = torch.cat([dense3_1,dense3_2],dim=1) # 16*512*32*32
      #   D3 = self.DS3(Concatenate3) # 16*512*16*16
        D3 = nn.MaxPool2d(kernel_size=(2,2))(Concatenate3)
      #   D3       = self.DS_maxpool(dense3_3)
                
        #print("D3: ",D3.shape)
      #   dense4_1 = conv_activation( self.d4_bn1(self.dense4_c1(D3))) # 16*512*16*16
      #   dense4_2 = conv_activation( self.d4_bn2(self.dense4_c2(dense4_1 ) )) # 16*512*16*16
        dense4_1 = conv_activation( self.d4_bn1( conv_activation(self.dense4_c1(D3)) ))  
        dense4_2 = conv_activation( self.dense4_c2(dense4_1) ) 
        Concatenate4 = torch.cat((dense4_1,dense4_2),dim=1) # 16*1024*16*16
        Concatenate4 = nn.Dropout(p=0.5)(Concatenate4)
      #   D4 = self.DS4(Concatenate4) # 16*1024*8*8
        D4 = nn.MaxPool2d(kernel_size=(2,2))(Concatenate4)
      #   D4       = self.DS_maxpool(dense4_3)

        #print("D4: ",D4.shape)
      #   dense5_1 = conv_activation( self.d5_bn1(self.dense5_c1(D4))) # 16*1024*8*8
      #   dense5_2 = conv_activation( self.d5_bn2(self.dense5_c2(dense5_1 ) )) # 16*1024*8*8
        dense5_1 = conv_activation( self.d5_bn1( conv_activation(self.dense5_c1(D4)) ))  
        dense5_2 = conv_activation( self.dense5_c2(dense5_1) ) 
        Concatenate5 = torch.cat((dense5_1,dense5_2),dim=1) # 16*2048*8*8
        Concatenate5 = nn.Dropout(p=0.5)(Concatenate5)
        
        #print("UP1: ",UP1.shape)
        UP1 = F.interpolate(Concatenate5,scale_factor=2,mode='nearest') # 16*2048*16*16
      #   dense6_1 = conv_activation( self.d6_bn1(self.dense6_c1(Concatenate5))) # 16*512*8*8
        dense6_1 = conv_activation(self.dense6_c1(UP1))  # 16*512*16*16
      #   UP1 = self.UP1(dense6_1)  # 16*512*16*16
        
      #   dense6_2 = conv_activation( self.d6_bn2(self.dense6_c2(torch.cat((Concatenate4,UP1),dim=1) ) )) # 16*1536*16*16 -> 16*512*16*16
      #   dense6_3 = conv_activation( self.d6_bn3(self.dense6_c3(dense6_2 ) )) # 16*512*16*16
        running_mean = self.d6_bn1.running_mean
        running_var = self.d6_bn1.running_var
        dense6_2 = conv_activation( self.d6_bn1( conv_activation(self.dense6_c2(torch.cat((Concatenate4,dense6_1),dim=1))) )) 
        dense6_3 = conv_activation( self.dense6_c3(dense6_2) ) 
        Concatenate6 = torch.cat((dense6_2,dense6_3),dim=1) # 16*1024*16*16


        #print("UP2: ",UP2.shape)
        UP2 = F.interpolate(Concatenate6,scale_factor=2,mode='nearest')
      #   dense7_1 = conv_activation( self.d7_bn1(self.dense7_c1(Concatenate6))) # 16*256*16*16
        dense7_1 = conv_activation(self.dense7_c1(UP2))
      #   UP2 = self.UP2(dense7_1) # 16*256*32*32
        
      #   dense7_2 = conv_activation( self.d7_bn2(self.dense7_c2(torch.cat((Concatenate3,UP2),dim=1) ) )) # 16*768*32*32 -> 16*256*32*32
      #   dense7_3 = conv_activation( self.d7_bn3(self.dense7_c3(dense7_2) )) # 16*256*32*32
        dense7_2 = conv_activation( self.d7_bn1( conv_activation(self.dense7_c2(torch.cat((Concatenate3,dense7_1),dim=1))) ))  
        dense7_3 = conv_activation( self.dense7_c3(dense7_2) ) 
        Concatenate7 = torch.cat((dense7_2,dense7_3),dim=1) # 16*512*32*32


        #print("UP3: ",UP3.shape)
        UP3 = F.interpolate(Concatenate7,scale_factor=2,mode='nearest')
      #   dense8_1 = conv_activation( self.d8_bn1(self.dense8_c1(Concatenate7))) # 16*128*32*32
        dense8_1 = conv_activation(self.dense8_c1(UP3))
      #   UP3 = self.UP3(dense8_1) # 16*128*64*64
        
      #   dense8_2 = conv_activation( self.d8_bn2(self.dense8_c2(torch.cat((Concatenate2,UP3),dim=1) ) )) # 16*384*64*64 -> 16*128*64*64
      #   dense8_3 = conv_activation( self.d8_bn3(self.dense8_c3(dense8_2 ) )) # 16*128*64*64
        dense8_2 = conv_activation( self.d8_bn1( conv_activation(self.dense8_c2(torch.cat((Concatenate2,dense8_1),dim=1))) ))  
        dense8_3 = conv_activation( self.dense8_c3(dense8_2) ) 
        Concatenate8 = torch.cat((dense8_2,dense8_3),dim=1) # 16*256*64*64
        

        #print("UP3: ",UP4.shape)
        UP4 = F.interpolate(Concatenate8,scale_factor=2,mode='nearest')
      #   dense9_1 = conv_activation( self.d9_bn1(self.dense9_c1(Concatenate8))) # 16*64*64*64
        dense9_1 = conv_activation(self.dense9_c1(UP4))
      #   UP4 = self.UP4(dense9_1) # 16*64*128*128
        
      #   dense9_2 = conv_activation( self.d9_bn2(self.dense9_c2(torch.cat((Concatenate1,UP4),dim=1) ) )) # 16*192*128*128 -> 16*64*128*128
      #   dense9_3 = conv_activation( self.d9_bn3(self.dense9_c3(dense9_2 ) )) # 16*64*128*128
        dense9_2 = conv_activation( self.d9_bn1( conv_activation(self.dense9_c2(torch.cat((Concatenate1,dense9_1),dim=1))) ))  
        dense9_3 = conv_activation( self.dense9_c3(dense9_2) ) 
        Concatenate9 = torch.cat((dense9_2,dense9_3),dim=1) # 16*128*128*128
        
        last_layer   = self.last_conv(Concatenate9) # 16*2*128*128
        # last_layer = self.FC(Concatenate9)
        #print(Mask.shape)
        output  = torch.multiply(last_layer,mask)+1e-5
        # output = last_layer
        return output


