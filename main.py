'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

# set seed 
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)

# set detect anomaly 
torch.autograd.set_detect_anomaly(True)

import wandb 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model_name', default='', type=str, help='model name')
parser.add_argument('--model_index', default=0, type=int, help='model index')
model_name_list = ['VGG19', 'ResNet18', 'PreActResNet18', 'GoogLeNet', 'DenseNet121', 'ResNeXt29_2x64d', 'MobileNet', 'MobileNetV2', 'DPN92', 'ShuffleNetG2', 'SENet18', 'ShuffleNetV2_1', 'EfficientNetB0', 'RegNetX_200MF', 'SimpleDLA']
args = parser.parse_args()
if args.model_name == '':
    print('model name is None')
    args.model_name = model_name_list[args.model_index]
    print('model name is set to {}'.format(args.model_name))
checkpoint_path = './checkpoint/{}/'.format(args.model_name)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

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


wandb.init(
    # set the wandb project where this run will be logged
    project="pytorch_models",
    # track hyperparameters and run metadata
    config=args,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


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
        
    wandb.log({"train_loss": train_loss/(batch_idx+1), "train_acc": 100.*correct/total})


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
            
    wandb.log({"test_loss": test_loss/(batch_idx+1), "test_acc": 100.*correct/total})

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        torch.save(state, checkpoint_path + 'ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()
