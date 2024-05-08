#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ImageDataset
from model import ResNet, ChannelAdjuster
from util import *


# 定义设备，优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # 加载预训练的 ResNet 模型
    resnet = ResNet(freeze_layers=True).to(device)
    channel_adjuster = ChannelAdjuster(input_channels=5).to(device)

    # Create an instance of the ImageDataset class
    dataset = ImageDataset(train_or_test='train')
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=4)

    # 设置模型为评估模式
    resnet.eval()

    # 通过 DataLoader 遍历数据
    for inputs_res, inputs_glcm, labels in tqdm(dataloader):
        inputs_res = inputs_res.to(device)
        inputs_glcm = inputs_glcm.to(device)
        # renet提取特征
        features_res = resnet(inputs_res).to(device)
        print("\n features_res:", features_res.shape)  # 输出特征的维度，通常是 (batch_size, 2048)
        # glcm提取计算灰度图像的纹理特征
        gray_imgs = tensor_to_grayscale_list(inputs_glcm)
        batch_result = batch_glcm(gray_imgs).to(device)
        # 对灰度图像的纹理特征进行特征提取
        features_glcm = resnet(channel_adjuster(batch_result)).to(device)
        print("features_glcm:", features_glcm.shape)  # 输出特征的维度，通常是 (batch_size, 2048)
