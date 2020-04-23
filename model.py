import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBn2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SEBlock(nn.Module):
    def __init__(self,in_channel,reduction=16):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel//reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channel//reduction, in_channel, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        se = self.pool(x)
        se = self.conv1(se)
        se = self.relu(se)
        se = self.conv2(se)
        x = x + self.sigmoid(se)
        return x


class SENextBottleneck(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, stride=1, group=32, reduction=16, excite_size=-1, is_shortcut=False):
        super(SENextBottleneck, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel,mid_channel, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(mid_channel,mid_channel, kernel_size=3, padding=1, stride=stride, groups=group)
        self.conv_bn3 = ConvBn2d(mid_channel,out_channel, kernel_size=1, padding=0, stride=1)
        self.se       = SEBlock(out_channel, reduction)
 
        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)
            

    def forward(self,x):
        residual = x
        b = F.relu(self.conv_bn1(x),inplace=True)
        b = F.relu(self.conv_bn2(b),inplace=True)
        b = self.se(self.conv_bn3(b))
        
        if self.is_shortcut:
            residual = self.shortcut(x)
            
        b += residual
        b = F.relu(b,inplace=True)
        return b

class SEResNext50(nn.Module):
    def __init__(self, num_class=6):
        super(SEResNext50, self).__init__()
        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True), #bias=0
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block0[0].bias.data.fill_(0.0)

        self.block1  = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2, ceil_mode=True),
             SENextBottleneck( 64, 128, 256, stride=1, is_shortcut=True,  excite_size=64),
          * [SENextBottleneck(256, 128, 256, stride=1, is_shortcut=False, excite_size=64) for i in range(1,3)],
        )

        self.block2  = nn.Sequential(
             SENextBottleneck(256, 256, 512, stride=2, is_shortcut=True,  excite_size=32),
          * [SENextBottleneck(512, 256, 512, stride=1, is_shortcut=False, excite_size=32) for i in range(1,4)],
        )

        self.block3  = nn.Sequential(
             SENextBottleneck( 512,512,1024, stride=2, is_shortcut=True,  excite_size=16),
          * [SENextBottleneck(1024,512,1024, stride=1, is_shortcut=False, excite_size=16) for i in range(1,6)],
        )

        self.block4 = nn.Sequential(
             SENextBottleneck(1024,1024,2048, stride=2, is_shortcut=True,  excite_size=8),
          * [SENextBottleneck(2048,1024,2048, stride=1, is_shortcut=False, excite_size=8) for i in range(1,3)],
        )

        self.logit = nn.Linear(2048,num_class)

    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit