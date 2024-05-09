#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import argparse
import wandb

from data.dataset import ImageDataset
from model import ResNet, SimpleRegressor
from util import *

# 定义设备，优先使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():

    parser = argparse.ArgumentParser(description='Camera based friction coefficient estimation using deep learning')
    
    parser.add_argument('--dropout_p', type=float, default=0.1, help="The dropout probability to use")   
    parser.add_argument('--batch_size', type=int, default=5, help="batch_size") 
    parser.add_argument('--max_epoch', type=int, default=30, help="max_epoch") 
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help="learning rate decay rate") 
    parser.add_argument('--experiment_name', type=str, default='testing', help='The name of the experiment to run')
    parser.add_argument('--use_wandb', action='store_true', default=False, help="If set, use wandb to keep track of experiments")
    
    args = parser.parse_args()
    return args

def train_one_epoch(model, optimizer, criterion, train_dataloader):
    
    model.train()
    # 加载预训练的 ResNet 模型和 ChannelAdjuster 模型
    resnet = ResNet(freeze_layers=True).to(device)
    resnet.eval()
    
    total_loss = 0

    # 通过 DataLoader 遍历数据
    for inputs_res, inputs_glcm, labels in tqdm(train_dataloader):
        inputs_res = inputs_res.to(device)
        inputs_glcm = inputs_glcm.to(device)
        labels = labels.to(device)
        
        # renet提取特征
        features_res = resnet(inputs_res).to(device)
        
        # glcm提取计算灰度图像的纹理特征
        gray_imgs = tensor_to_grayscale_list(inputs_glcm)
        batch_result = batch_glcm(gray_imgs).to(device)
        # 对灰度图像的纹理特征进行特征提取
        features_glcm = resnet(batch_result).to(device)
        
        input = torch.cat((features_res, features_glcm), dim=1)
        
        optimizer.zero_grad()
        outputs = model(input)
        outputs = outputs.squeeze(1)
        # print(outputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input.size(0)
        
    return total_loss / len(train_dataloader.dataset)


def evaluate_one_epoch(model, criterion, dev_dataloader):
        
    model.eval()
    resnet = ResNet(freeze_layers=True).to(device)
    resnet.eval()
    
    total_loss = 0

    with torch.no_grad():
        for inputs_res, inputs_glcm, labels in tqdm(dev_dataloader):
            inputs_res = inputs_res.to(device)
            inputs_glcm = inputs_glcm.to(device)
            labels = labels.to(device)
            
            features_res = resnet(inputs_res).to(device)
            features_glcm = resnet(inputs_glcm).to(device)
            input = torch.cat((features_res, features_glcm), dim=1)
            outputs = model(input)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * input.size(0)
        
    return total_loss / len(dev_dataloader.dataset)


def main():
        
    args = get_args()
    
    if args.use_wandb:
        setup_wandb(args, args.experiment_name)
        
    dropout_p = args.dropout_p
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    lr = args.lr
    lr_decay = args.lr_decay
    experiment_name = args.experiment_name
    
    checkpoint_dir = os.path.join(f'model/{experiment_name}/')
    # Check if the directory exists
    if not os.path.exists(checkpoint_dir):
        # Create the directory if it doesn't exist
        os.makedirs(checkpoint_dir)
    
    # 创建 DataLoader
    train_dataloader = DataLoader(ImageDataset(train_or_test='train'), batch_size=batch_size, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(ImageDataset(train_or_test='dev'), batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = SimpleRegressor(dropout_p=dropout_p).to(device)
    
    print(f"Model is training on {next(model.parameters()).device} !!!")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=lr_decay)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    
    for epoch in range(max_epoch):
        
        train_loss = train_one_epoch(model, optimizer, criterion, train_dataloader)
        dev_loss = evaluate_one_epoch(model, criterion, dev_dataloader)
        
        print(f"Epoch {epoch+1}/{max_epoch}:")
        print(f"Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")
        
        scheduler.step()
        
        # 存储检查点
        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir,'best.pth'))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir,f'{epoch+1}.pth'))
            print(f'Checkpoint saved at Epoch {epoch+1}')
        
        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "dev_loss": dev_loss, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch+1})
            
    if args.use_wandb:
        wandb.finish()
        
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir,'best.pth')))
    dev_loss = evaluate_one_epoch(model, criterion, dev_dataloader)
    print("Final Dev Loss: ", dev_loss)


if __name__ == "__main__":
    main()

