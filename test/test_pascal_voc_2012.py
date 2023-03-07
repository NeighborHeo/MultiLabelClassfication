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
    # from Image PIL to Tensor
     
    # transforms.Resize(img_size),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    # transforms.Resize(img_size),
    transforms.Resize((256, 256)),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def test_load_pascal_voc_2012():
    trainset = torchvision.datasets.VOCDetection(
       root='~/.data', year='2012', image_set='train', download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2)
    testset = torchvision.datasets.VOCDetection(
        root='~/.data', year='2012', image_set='val', download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2)
    classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    num_classes = 20
    
    
    for i, (image, target) in enumerate(trainset):
        print(image)
        print(target)
        break
        
    return trainloader, testloader, classes, num_classes

def main():
    trainloader, testloader, classes, num_classes = test_load_pascal_voc_2012()
    print('trainloader: ', trainloader)
    print('testloader: ', testloader)
    print('classes: ', classes)
    print('num_classes: ', num_classes)
        # data, target = data
        # print('data: ', data)
    
    print('Done!')
    
if __name__ == '__main__':
    main()
    