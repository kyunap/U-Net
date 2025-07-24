import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import random
import math
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from albumentations.augmentations import transforms
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations.core.composition import Compose, OneOf

from PIL import Image

# 사용자 정의
import preprocess
import utils
import models
import evaluation as ev

# custom dataset 
class Dataset(Dataset): 
 
    def __init__(self, data_path, mode =None, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        
        """tiff 파일 읽기"""
        self.inputs = utils.read_tiff(self.data_path+'train-input.tif')
        self.labels = utils.read_tiff(self.data_path+'train-labels.tif')

        """train/val 분리"""
        idx = utils.set_index(len(self.inputs), 0.7, self.mode)
        self.inputs = self.inputs[idx]
        self.labels = self.labels[idx]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs = self.inputs[index]
        labels = self.labels[index]
        
        inputs = inputs/255
        labels = labels/255
        
        """gray image channel 추가"""
        if labels.ndim == 2:  
            labels = np.expand_dims(labels, -1)
        if inputs.ndim == 2:  
            inputs = np.expand_dims(inputs, -1)
        
        """augmentation 적용"""
        if self.transform is not None:
            augmented = self.transform(image = inputs, mask = labels)
            inputs, labels = augmented['image'],augmented["mask"]
            inputs = torch.from_numpy(inputs).permute(2,0,1)
            labels = torch.from_numpy(labels).permute(2,0,1)
 
        return inputs, labels
    
    def make_transform(mode):
        if mode == 'train':
            train_transform = Compose([
             transforms.Resize(height = 512, width = 512),
            OneOf([transforms.MotionBlur(),
               transforms.OpticalDistortion(),
                transforms.GaussNoise(p = 0.5),
                transforms.RandomContrast()]),
            transforms.ElasticTransform(),
            OneOf([transforms.HorizontalFlip(),
             transforms.RandomRotate90(),
             transforms.VerticalFlip()]), # oneof에 p 부여 가능,
             transforms.Normalize((0.5), (0.5))
             ])        
            return train_transform
        else:
            test_transform = Compose([
     transforms.Resize(height = 512, width = 512),
     transforms.Normalize((0.5), (0.5))
     ])
        return test_transform
    
    class DoubleConv(nn.Module):
        """반복되는 conv - BN - ReLU 구조 모듈화"""
        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    """가중치 초기화"""
def weights_init(init_type='xavier'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (classname.find('Norm') == 0):
            if hasattr(m, 'weight') and m.weight is not None:
                init.constant_(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun

# 파라미터 설정
batch_size = 5
epochs = 200
learning_rate = 0.001

# augmentation
train_transform = preprocess.make_transform('train')
test_transform = preprocess.make_transform('test')

# data load
train = preprocess.Dataset(data_path, 'train', transform=train_transform)
val = preprocess.Dataset(data_path, 'val', transform = test_transform)

train_loader = DataLoader(train, batch_size = batch_size,shuffle=True)
val_loader = DataLoader(val, batch_size = batch_size,shuffle=False)

device = 'cuda:1'
model = models.UNet(1, 1).to(device)

""" optimizer: RMSprop 및 Adam """
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-8)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=30, factor= 0.1)
criterion = nn.BCEWithLogitsLoss()  

"""가중치 초기값 적용"""
init_weights = preprocess.weights_init('kaming')
model.apply(init_weights)

def train_model(model,train_loader, epochs,device,optimizer, scheduler, criterion, model_path,val_loader=None): 
    score_dict = {}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
    
        for i, (images, masks) in enumerate(train_loader):
            
            imgs = images
            true_masks = masks
    
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if model.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
    
            masks_pred = model(imgs)
            loss = criterion(masks_pred, true_masks)
            train_loss += loss.item()
    
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()
            
            """검증셋 있을 경우만 사용"""
            if val_loader != None:
                   
                model.eval()
                val_loss = 0 
                for j, (images, masks) in enumerate(val_loader):
                
                    imgs = images
                    true_masks = masks
            
                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if model.n_classes == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
            
                    masks_pred = model(imgs)
                    loss = criterion(masks_pred, true_masks)
                    val_loss += loss.item()              
                
            else:
                val_loss = 0
                j = 0
        #if val_loader == None:
            #schedule_standard = train_loss/(i+1)
        #else:
            #schedule_standard = val_loss/(j+1)
        
        """Scheduler update"""
        schedule_standard = train_loss/(i+1)    
        scheduler.step(schedule_standard)       
        print("epoch: {}/{}  | trn loss: {:.4f} | val loss: {:.4f}".format(
            epochs, epoch+1, train_loss /(i+1), val_loss /(j+1)))   
        score_dict[epoch] = {'train':train_loss/(i+1), 'val':val_loss/(j+1)}

        """Model save"""
        checkpoint = {'loss':train_loss/(i+1),
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, model_path+'{}_epoch.pth'.format(epoch))  
    
    return model, score_dict

def eval_model(model, loader,device):

    model.eval()
    with torch.no_grad():
        mask_type = torch.float32 if model.n_classes == 1 else torch.long
        total_loss = 0
        iou_score = 0
        preds = []
        preds_thres = []
        labels = []
        
        for j, (images, masks) in enumerate(loader):
            imgs, true_masks = images, masks
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            masks_pred = model(imgs)

            if model.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                criterion = nn.BCEWithLogitsLoss()
                pred = masks_pred
                loss = criterion(masks_pred, true_masks)
                
                pred2 = torch.sigmoid(masks_pred)
                pred2 = (pred > 0.5).float()
                
                pred = pred.cpu().numpy()
                pred2 = pred2.cpu().numpy()
                true_masks = true_masks.cpu().numpy()
                
                preds.append(pred)
                preds_thres.append(pred2)
                labels.append(true_masks)
                total_loss += loss.item()
                """IOU Score"""
                iou_score += compute_iou(pred2, true_masks)
 
    return np.vstack(preds), np.vstack(preds_thres), np.vstack(labels), (total_loss/(j+1), iou_score/(j+1))