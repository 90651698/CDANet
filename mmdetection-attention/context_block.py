import torch
from torch import nn
import torch.nn.functional as F
from ..utils import constant_init, kaiming_init
from .registry import PLUGIN_LAYERS


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


@PLUGIN_LAYERS.register_module()
class ContextBlock(nn.Module):

    _abbr_ = 'context_block'

    def __init__(self,
                 in_channels,
                 ratio
                 ):
        super(ContextBlock, self).__init__()
        
        ################Channel
        self.conv1=nn.Conv2d(64,32,kernel_size=1,padding=0, bias=False)
        self.conv2=nn.Conv2d(128,64,kernel_size=1,padding=0, bias=False)
        self.conv3=nn.Conv2d(256,128,kernel_size=1,padding=0, bias=False)
        self.conv4=nn.Conv2d(512,256,kernel_size=1,padding=0, bias=False)
        self.conv5=nn.Conv2d(1024,512,kernel_size=1,padding=0, bias=False)
        self.conv6=nn.Conv2d(2048,1024,kernel_size=1,padding=0, bias=False)
        
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv7=nn.Conv1d(1,1,kernel_size=3,padding=1, bias=False)#padding=int(kernel_size/2)
        self.conv8=nn.Conv1d(1,1,kernel_size=5,padding=2, bias=False)
        self.conv9=nn.Conv1d(1,1,kernel_size=7,padding=3, bias=False)

        self.sigmoid = nn.Sigmoid()

        ###########Spatial
        self.convs1=nn.Conv2d(64,1,kernel_size=1,padding=0, bias=False)
        self.convs2=nn.Conv2d(128,1,kernel_size=1,padding=0, bias=False)
        self.convs3=nn.Conv2d(256,1,kernel_size=1,padding=0, bias=False)
        self.convs4=nn.Conv2d(512,1,kernel_size=1,padding=0, bias=False)
        self.convs5=nn.Conv2d(1024,1,kernel_size=1,padding=0, bias=False)
        self.convs6=nn.Conv2d(2048,1,kernel_size=1,padding=0, bias=False)


        # k_size = 7
        self.spatial = nn.Sequential(
            nn.Conv2d(
            2,
            1,
            kernel_size=7,
            stride=1,
            padding=3,
            dilation=1,
            groups=1,#每个卷积核的通道数in_planes/groups
            # relu=True,
            # bn=True,
            bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU() #激活函数
        )

    def ChannelGate(self,x):
        N,C,H,W=x.size()
        # print("channel-x",x.shape)
        global B1
        global B2
        global D1
        global D2
        global E1
        global E2

        if C == 64 :
            B1=self.conv1(x)
            B2=self.conv1(x)
        elif C==128 :
            B1=self.conv2(x)
            B2=self.conv2(x)
        elif C==256 :
            B1=self.conv3(x)
            B2=self.conv3(x)
        elif C==512 :
            B1=self.conv4(x)
            B2=self.conv4(x)
        elif C==1024:
            B1=self.conv5(x)
            B2=self.conv5(x)
        elif C==2048:
            B1=self.conv6(x)
            B2=self.conv6(x)
        else:
            print("channel-C",C)
        # print("channel-B1",B1.shape)
        # print("channel-B2",B2.shape)


        D1=(B1+B2)/2
        D2=(B1-B2)/2
        # print("channel-D1",D1.shape)
        # print("channel-D2",D2.shape)

        avg_pool1=self.avg_pool(D1)
        avg_pool2=self.avg_pool(D2)
        # print("channel-avg_pool1",avg_pool1.shape)
        # print("channel-avg_pool2",avg_pool2.shape)

        if C == 64:
            E1=self.conv6(avg_pool1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            E2=self.conv6(avg_pool2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        elif C==128 or C==256 or C==512: #C=128,256,512,k=5
            E1=self.conv7(avg_pool1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            E2=self.conv7(avg_pool2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        elif C==1024 or C==2048:
            E1=self.conv8(avg_pool1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            E2=self.conv8(avg_pool2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # else:
        #     print("C",C)
        # print("channel-E1",E1.shape)
        # print("channel-E2",E2.shape)

        channel_att = torch.cat(
            (E1, E2),
            dim=1,
        )
        # print("channel_att",channel_att.shape)

        scale = (
            F.sigmoid(channel_att)
        )
        # print("channel-scale",scale.shape)
        # print("channel-x*scale",(x*scale).shape)
        
        return (x*scale)
    def SpatialGate(self,x):
        N,C,H,W=x.size()
        global F1
        global F2
        global H1
        global H2

        if C == 64 :
            F1=self.convs1(x)
            F2=self.convs1(x)

        elif C==128 :
            F1=self.convs2(x)
            F2=self.convs2(x)
        elif C==256 :
            F1=self.convs3(x)
            F2=self.convs3(x)

        elif C==512 :
            F1=self.convs4(x)
            F2=self.convs4(x)
        # print("B1",B1.shape)
        elif C==1024 :
            F1=self.convs5(x)
            F2=self.convs5(x)
        elif C==2048 :
            F1=self.convs6(x)
            F2=self.convs6(x)
        else:
            print("spatial-C",C)

        H1=(F1+F2)/2
        H2=(F1-F2)/2
    
        # print("D1",D1.shape)
        avg1=torch.mean(H1,1)
        avg2=torch.mean(H2,1)
        avg1=torch.unsqueeze(avg1,1)
        avg2=torch.unsqueeze(avg2,1)
        # print("avg1",avg1.shape)

        x_compress=torch.cat(
            (avg1, avg2),
            dim=1,
        )
        # print("x_com",x_compress.shape)
        y = self.spatial(x_compress)
        scale = F.sigmoid(y)


        return x*scale


    def forward(self,x):
        out= self.ChannelGate(x)
        # out= self.SpatialGate(x)

        out= out+ self.SpatialGate(x)
        return out
