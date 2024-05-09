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


def test_one_epoch(model, criterion, dataloader):
        
    model.eval()
    resnet = ResNet(freeze_layers=True)
    resnet.eval()
    
    total_loss = 0

    with torch.no_grad():
        for i, (inputs_res, inputs_glcm, labels) in enumerate(dataloader):
            inputs_res = inputs_res
            inputs_glcm = inputs_glcm
            labels = labels
            
            features_res = resnet(inputs_res)
            features_glcm = resnet(inputs_glcm)
            input = torch.cat((features_res, features_glcm), dim=1)
            outputs = model(input)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            print(f"No:{i:3d},Estimated: {outputs.item():.4f}, Ground Truth: {labels.item():.4f}, Loss: {loss.item():.4f}")
            total_loss += loss.item() * input.size(0)
        
    return total_loss / len(dataloader.dataset)


def main():
        
    # 创建 DataLoader
    test_dataloader = DataLoader(ImageDataset(train_or_test='test'), batch_size=1, shuffle=True, num_workers=4)
    
    model = SimpleRegressor()
    
    criterion = nn.MSELoss()  
    
    model.load_state_dict(torch.load(os.path.join('model','official_train','best.pth')))

    test_loss = test_one_epoch(model, criterion, test_dataloader)
    
    print("Final test Loss: ", test_loss)
    
    
if __name__ == "__main__":
    main()
    