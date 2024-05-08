#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from dataset import ImageDataset
from util import ResNet

# 定义设备，优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":


    # 加载预训练的 ResNet 模型
    resnet = ResNet(freeze_layers=True).to(device)

    # Create an instance of the ImageDataset class
    dataset = ImageDataset(train_or_test='train')
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)

    # 设置模型为评估模式
    resnet.eval()

    # 通过 DataLoader 遍历数据
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # 提取特征
        features = resnet(inputs)
        print(features.shape)  # 输出特征的维度，通常是 (batch_size, 2048)

