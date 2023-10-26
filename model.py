import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups,
                     bias=False, dilation=dilation)

"""basic block"""
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None):
        super(BasicBlock,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)

            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x):
        identity = x
        #first layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #second layer
        out = self.conv2(out)
        out = self.bn2(out)

        #
        if self.downsample is not None:
            identity = self.downsample(x)
            out += identity
            out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = BasicBlock(inplanes=64, planes=64)
        self.layer2 = BasicBlock(inplanes=64, planes=128, stride=2, downsample=nn.Conv2d(64,128, kernel_size=1, stride=2))
        self.layer3 = BasicBlock(inplanes=128, planes=256, stride=2, downsample=nn.Conv2d(128, 256, kernel_size=1, stride=2))
        self.layer4 = BasicBlock(inplanes=256, planes=256, stride=2, downsample=nn.Conv2d(256, 256, kernel_size=1, stride=2))
        self.pool5 = nn.MaxPool2d(kernel_size=7)
        self.fc6 = nn.Linear(256, 100)

        self.res = nn.Softmax()
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool5(out)
        out = out.view(-1, out.size(0))
        out = self.fc6(out)
        out = self.res(out)
        return out

