import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *




def SelectModel(m):
    
    if m == 'Resnet29':
        return ResNet(Bottleneck, [3, 3, 3])
    elif m == 'Resnext29-32x4d':
        return ResNeXt(ResNextBlock, [3, 3, 3], 32, 4)
    elif m == 'Resnext29-64x4d':
        return ResNeXt(ResNextBlock, [3, 3, 3], 64, 4)




class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

        

        #weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.layer1(o)
        o = self.layer2(o)
        o = self.layer3(o)
        o = F.avg_pool2d(o, o.size()[3])
        o = o.view(o.size(0), -1)
        o = self.linear(o)

        return o


class ResNeXt(nn.Module):
    def __init__(self, block, num_blocks, cardinality, width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.in_planes = 64
        self.width = width

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], cardinality, self.width, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], cardinality, self.width,stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], cardinality, self.width,stride=2)
        self.linear = nn.Linear(block.expansion*256, num_classes)

        

        #weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, planes, num_blocks, cardinality, width, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes, planes, cardinality, self.width, stride))
            self.in_planes = block.expansion*planes
            
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.layer1(o)
        o = self.layer2(o)
        o = self.layer3(o)
        o = F.avg_pool2d(o, o.size()[3])
        o = o.view(o.size(0), -1)
        o = self.linear(o)

        return o
    
