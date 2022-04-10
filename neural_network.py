import torch
import torch.nn as nn

first_HL = 256

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class SpinalResNet(nn.Module):

    def __init__(self, block, duplicates, num_classes=10):
        super(SpinalResNet, self).__init__()

        self.in_channels = 32
        self.conv1 = conv3x3(in_channels=3, out_channels=32)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)

        self.conv2_x = self._make_block(block, duplicates[0], out_channels=32)
        self.conv3_x = self._make_block(block, duplicates[1], out_channels=64, stride=2)
        self.conv4_x = self._make_block(block, duplicates[2], out_channels=128, stride=2)
        self.conv5_x = self._make_block(block, duplicates[3], out_channels=256, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(256, first_HL) 
        self.fc1_1 = nn.Linear(256 + first_HL, first_HL)
        self.fc1_2 = nn.Linear(256 + first_HL, first_HL)
        self.fc1_3 = nn.Linear(256 + first_HL, first_HL)
        
        self.fc_layer = nn.Linear(first_HL*4, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_block(self, block, duplicates, out_channels, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, duplicates):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        
        
        out1 = self.maxpool2(out)
        out2 = out1[:,:,0,0]
        out2 = out2.view(out2.size(0),-1)
        
        x1 = out1[:,:,0,0]
        x1 = self.relu(self.fc1(x1))
        x2= torch.cat([ out1[:,:,0,1], x1], dim=1)
        x2 = self.relu(self.fc1_1(x2))
        x3= torch.cat([ out1[:,:,1,0], x2], dim=1)
        x3 = self.relu(self.fc1_2(x3))
        x4= torch.cat([ out1[:,:,1,1], x3], dim=1)
        x4 = self.relu(self.fc1_3(x4))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        out = torch.cat([x, x4], dim=1)
        
        out = self.fc_layer(out)

        return out

def SpinalResNet18():
    return SpinalResNet(BasicBlock, [2,2,2,2])

def SpinalResNet34():
    return SpinalResNet(BasicBlock, [3, 4, 6, 3])