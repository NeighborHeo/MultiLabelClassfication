# %% 
'''test_pasca_voc_2012.py'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from PIL import Image

img_size = (224, 224)

transform_train = transforms.Compose([
    # transforms.Resize(img_size),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    # transforms.Resize(img_size),
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class myCustomDataset(torch.utils.data.Dataset):
    
    def __init__(self, imgs, targets, train=True, singlelabel=True, transform=None, target_transform=None):
        self.imgs = imgs
        self.targets = targets
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        if singlelabel:
            self.imgs, self.targets = self.erase_multi_labels(self.imgs, self.targets)
        self.targets = np.argmax(self.targets, axis=1)
            
    def erase_multi_labels(self, imgs, targets):
        sum_labels = np.sum(targets, axis=1)
        index = np.where(sum_labels == 1)
        imgs = imgs[index]
        targets = targets[index]
        return imgs, targets
    
    def __getitem__(self, index):
        imgs, targets = self.imgs[index], self.targets[index]
        if self.transform is not None:
            imgs = self.transform(imgs)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return imgs, targets
    
    def __len__(self):
        return len(self.imgs)
     

def test_load_pascal_voc_2012():
    
        
    trainset = torchvision.datasets.CIFAR10(
        root='~/.data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='~/.data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    import pathlib
    datapath = "/home/suncheol/.data/PASCAL_VOC_2012/"
    path = pathlib.Path(datapath)
    test_imgs = np.load(path.joinpath('PASCAL_VOC_val_224_Img.npy'))
    test_imgs = test_imgs.transpose(0, 2, 3, 1)*255
    test_imgs = test_imgs.astype(np.uint8)
    
    test_labels = np.load(path.joinpath('PASCAL_VOC_val_224_Label.npy'))
    print("size of test dataset: ", test_imgs.shape, "images")
    
    # train_imgs = np.load(path.joinpath('PASCAL_VOC_train_224_Img.npy'))
    flpath = path.joinpath('dirichlet', 'alpha_1')
    nth = 0
    train_imgs = np.load(flpath.joinpath(f'Party_{nth}_X_data.npy'))
    train_imgs = train_imgs.transpose(0, 2, 3, 1)*255
    train_imgs = train_imgs.astype(np.uint8)
    train_labels = np.load(flpath.joinpath(f'Party_{nth}_y_data.npy'))
    print("size of train dataset: ", train_imgs.shape, "images")
    
    train_dataset = myCustomDataset(train_imgs, train_labels, train=True, singlelabel=True, transform=transform_train, target_transform=None)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    
    for i, (image, target) in enumerate(train_loader):
    # for i, (image, target) in enumerate(train_dataset):
        print(image)
        print(target)
        print(image.shape)
        print(target.shape)
        break
    
    sum_labels = np.sum(train_labels, axis=1)
    print(len(sum_labels))
    index = np.where(sum_labels == 1)
    train_labels = train_labels[index]
    train_imgs = train_imgs[index]
    print(train_imgs.shape)
     
    # eject multi class label 
    # test_labels # (4952, 20 ; len, num_classes)
    sum_labels = np.sum(test_labels, axis=1)
    print(len(sum_labels))
    index = np.where(sum_labels == 1)
    test_labels = test_labels[index]
    test_imgs = test_imgs[index]
    print(test_imgs.shape)
     
def main():
    test_load_pascal_voc_2012()

# def test_load_pascal_voc_2012():
#     trainset = torchvision.datasets.VOCDetection(
#        root='~/.data', year='2012', image_set='train', download=False, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=64, shuffle=True, num_workers=2)
#     testset = torchvision.datasets.VOCDetection(
#         root='~/.data', year='2012', image_set='val', download=False, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=64, shuffle=False, num_workers=2)
#     classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
#             'bus', 'car', 'cat', 'chair', 'cow',
#             'diningtable', 'dog', 'horse', 'motorbike', 'person',
#             'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
#     num_classes = 20
    
#     current_time = time.time()
#     for i, (image, target) in enumerate(trainset):
#         # print(image)
#         # print(target)
#         # print(image.shape)
#         print(target)

#     print('time: ', time.time() - current_time)
        
#     return trainloader, testloader, classes, num_classes

# def main():
#     trainloader, testloader, classes, num_classes = test_load_pascal_voc_2012()
#     print('trainloader: ', trainloader)
#     print('testloader: ', testloader)
#     print('classes: ', classes)
#     print('num_classes: ', num_classes)
#         # data, target = data
#         # print('data: ', data)
    
#     print('Done!')
    
if __name__ == '__main__':
    main()
    
# %%
