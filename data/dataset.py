#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import torch
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, transform=None, train_or_test='train', if_resize=True):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if train_or_test == 'train':
            csv_file_path = os.path.join(script_dir, 'train.csv')
        else:
            csv_file_path = os.path.join(script_dir, 'test.csv')
        self.data = pd.read_csv(csv_file_path)
        self.data = self.data.iloc[1:]  # 移除首行(表头)

        # 如果没有特定的转换被提供，就使用默认的转换（即转换到张量）
        if transform is None:
            if if_resize:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),  # 调整图像大小
                    transforms.ToTensor(),  # 添加这行来转换图像到张量
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),  # 添加这行来转换图像到张量
                ])
        else:
            self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(os.path.dirname(__file__), "Images", img_name)
        gt_value = self.data.iloc[idx, 1]
        
        img = Image.open(img_path).convert("RGB")

        # 应用转换
        if self.transform:
            img = self.transform(img)

        # 将 gt_value 转换为张量
        gt_value = torch.tensor(gt_value, dtype=torch.float32)  # 使用 float32 类型，如果是分类任务，可能需要 torch.long

        return img, gt_value

    
if __name__ == "__main__":

    # Create an instance of the ImageDataset class
    dataset = ImageDataset(train_or_test='train')
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)

    # 使用 DataLoader
    for images, gt_values in dataloader:
        # 这里可以进行你的模型训练等操作
        print(images.shape, gt_values.shape)
        
        
    # Create an instance of the ImageDataset class
    dataset = ImageDataset(train_or_test='test')
    
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4)

    # 使用 DataLoader
    for images, gt_values in dataloader:
        # 这里可以进行你的模型训练等操作
        print(images.shape, gt_values.shape)
    

    # # Access the elements of the dataset
    # for i in range(len(dataset)):
    #     img, gt_value = dataset[i]
    #     # Do something with the image and ground truth value
    #     print(f"No:{i},Image: {img}, Ground Truth Value: {gt_value}")
