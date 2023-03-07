'''Train CIFAR10 with PyTorch.'''
import os
import pathlib
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from functools import partial

import timm
from models import *
import models.vit_models as vit_models

import utils
from utils.utils import progress_bar
from utils.earlystop import earlystop
from utils.custom_dataset import myCustomDataset

utils.set_seed(0)

# set detect anomaly 
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model_name', default='', type=str, help='model name')
parser.add_argument('--model_index', default=0, type=int, help='model index')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--earlystop', action='store_true', help='use early stop')
parser.add_argument('--nindex', default=-1, type=int, help='index of client')
model_name_list = ['VGG19', 'ResNet18', 'PreActResNet18', 'GoogLeNet', 'DenseNet121', 'ResNeXt29_2x64d', 'MobileNet', 'MobileNetV2', 'DPN92', 'ShuffleNetG2', 'SENet18', 'ShuffleNetV2_1', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA', 'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224', 'head1', 'head2', 'head3']
args = parser.parse_args()
if args.model_name == '':
    print('model name is None')
    args.model_name = model_name_list[args.model_index]
    
print('model name is {}'.format(args.model_name))
checkpoint_path = './checkpoint/{}_{}_{}_{}/'.format(args.dataset, args.model_name, args.lr, args.nindex)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

img_size = 32
if args.model_name == 'vit_tiny_patch16_224' or args.model_name == 'vit_small_patch16_224' or args.model_name == 'vit_base_patch16_224':
    img_size = 224
elif args.model_name == 'head1' or args.model_name == 'head2' or args.model_name == 'head3':
    img_size = 224

# Data
print('==> Preparing data..')

# cifar10
if args.dataset == 'cifar10':

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='~/.data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='~/.data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = 10

elif args.dataset == 'pascal_voc':
    # pascal voc

    transform_train = transforms.Compose([
        # transforms.Resize(img_size),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomGrayscale(p=0.2),
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

    datapath = "/home/suncheol/.data/PASCAL_VOC_2012/"
    path = pathlib.Path(datapath)
    test_imgs = np.load(path.joinpath('PASCAL_VOC_val_224_Img.npy'))
    test_imgs = test_imgs.transpose(0, 2, 3, 1)*255
    test_imgs = test_imgs.astype(np.uint8)
    test_labels = np.load(path.joinpath('PASCAL_VOC_val_224_Label.npy'))
    sum_labels = np.sum(test_labels, axis=1)
    index = np.where(sum_labels == 1)
    test_labels = test_labels[index]
    test_imgs = test_imgs[index]
    print("size of test dataset: ", test_imgs.shape, "images")

    if args.nindex == -1:
        train_imgs = np.load(path.joinpath('PASCAL_VOC_train_224_Img.npy'))
        train_imgs = train_imgs.transpose(0, 2, 3, 1)*255
        train_imgs = train_imgs.astype(np.uint8)
        train_labels = np.load(path.joinpath('PASCAL_VOC_train_224_Label.npy'))
        print("size of train dataset: ", train_imgs.shape, "images")
    else:
        nth = args.nindex
        flpath = path.joinpath('dirichlet', 'alpha_1')
        train_imgs = np.load(flpath.joinpath(f'Party_{nth}_X_data.npy'))
        train_imgs = train_imgs.transpose(0, 2, 3, 1)*255
        train_imgs = train_imgs.astype(np.uint8)
        train_labels = np.load(flpath.joinpath(f'Party_{nth}_y_data.npy'))
        print("size of train dataset: ", train_imgs.shape, "images")

    sum_labels = np.sum(train_labels, axis=1)
    index = np.where(sum_labels == 1)
    train_labels = train_labels[index]
    train_imgs = train_imgs[index]

    train_dataset = myCustomDataset(train_imgs, train_labels, train=True, singlelabel=True, transform=transform_train, target_transform=None)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

    test_dataset = myCustomDataset(test_imgs, test_labels, train=False, singlelabel=True, transform=transform_test, target_transform=None)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    num_classes = 20

# eject multi class label 
# test_labels # (4952, 20 ; len, num_classes)

# Model
print('==> Building model..')

if args.model_name == 'VGG19':
    net = VGG('VGG19')
elif args.model_name == 'ResNet18':
    net = ResNet18()
elif args.model_name == 'PreActResNet18':
    net = PreActResNet18()
elif args.model_name == 'GoogLeNet':
    net = GoogLeNet()
elif args.model_name == 'DenseNet121':
    net = DenseNet121()
elif args.model_name == 'ResNeXt29_2x64d':
    net = ResNeXt29_2x64d()
elif args.model_name == 'MobileNet':
    net = MobileNet()
elif args.model_name == 'MobileNetV2':
    net = MobileNetV2()
elif args.model_name == 'DPN92':
    net = DPN92()
elif args.model_name == 'ShuffleNetG2':
    net = ShuffleNetG2()
elif args.model_name == 'SENet18':
    net = SENet18()
elif args.model_name == 'ShuffleNetV2_1':
    net = ShuffleNetV2(1)
elif args.model_name == 'EfficientNetB0':
    net = EfficientNetB0()
elif args.model_name == 'RegNetX_200MF':
    net = RegNetX_200MF()
elif args.model_name == 'SimpleDLA':
    net = SimpleDLA()
elif args.model_name == 'vit_tiny_patch16_224':
    net = timm.create_model('vit_tiny_patch16_224', pretrained=args.pretrained)
    net.head = nn.Linear(net.head.in_features, num_classes)
elif args.model_name == 'vit_small_patch16_224':
    net = timm.create_model('vit_small_patch16_224', pretrained=args.pretrained)
    net.head = nn.Linear(net.head.in_features, num_classes)
elif args.model_name == 'vit_base_patch16_224':
    net = timm.create_model('vit_base_patch16_224', pretrained=args.pretrained)
    net.head = nn.Linear(net.head.in_features, num_classes)
elif args.model_name == 'head1':
    net = vit_models.VisionTransformer(
        img_size=224, patch_size=16, num_classes=num_classes, num_heads=1, embed_dim=192, depth=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
elif args.model_name == 'head2':
    net = vit_models.VisionTransformer(
        img_size=224, patch_size=16, num_classes=num_classes, num_heads=2, embed_dim=192, depth=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
elif args.model_name == 'head3':
    net = vit_models.VisionTransformer(
        img_size=224, patch_size=16, num_classes=num_classes, num_heads=3, embed_dim=192, depth=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
else:
    raise ValueError('No such model')

# net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_path + 'ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="3JenmgUXXmWcKcoRk8Yra0XcD",
    project_name="pytorch-model",
    workspace="neighborheo",
)

experiment.log_parameters(args)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if args.earlystop:
    early_stopping = earlystop(save_path=checkpoint_path, verbose=True)
else:
    early_stopping = earlystop(save_path=checkpoint_path, patience=1000, verbose=True)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    metrics = { "train_loss": train_loss/(batch_idx+1), "train_acc": 100.*correct/total }
    experiment.log_metrics(metrics, step=epoch)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    metrics = { "test_loss": test_loss/(batch_idx+1), "test_acc": 100.*correct/total }
    experiment.log_metrics(metrics, step=epoch)
    early_stopping(test_loss/(batch_idx+1), net)

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
    if early_stopping.stop():
        break
