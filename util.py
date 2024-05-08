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


if __name__ == "__main__":
    
    # 实例化模型
    model = ResNet(freeze_layers=True)

    # 打印模型结构
    print(model)
