#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, freeze_layers=True):
        super(ResNet, self).__init__()
        # 加载预训练的 ResNet50 模型
        base_model = models.resnet50(pretrained=True)
        
        # 如果决定冻结预训练层的权重
        if freeze_layers:
            for param in base_model.parameters():
                param.requires_grad = False
        
        # 移除原有的全连接层
        base_model.fc = nn.Identity()
        
        self.resnet = base_model

    def forward(self, x):
        # 通过 ResNet 获取特征
        features = self.resnet(x)
        return features
    
class ChannelAdjuster(nn.Module):
    def __init__(self, input_channels):
        super(ChannelAdjuster, self).__init__()
        # 定义一个卷积层，输入通道为input_channels，输出通道为3
        # 这里的kernel_size设为1，使得这个转换不会改变空间维度
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        # 应用卷积层，x的形状应该是(B, C, W, H)
        x = self.conv(x)
        return x


if __name__ == "__main__":
    
    # 实例化模型
    model = ResNet(freeze_layers=True)

    # 打印模型结构
    print(model)
