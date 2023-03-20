import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import timm
import os

# 데이터셋을 불러오고 전처리를 수행합니다.
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

import os
import urllib.request
import zipfile

# CelebA 데이터셋을 다운로드하고 압축을 해제합니다.
url = 'https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1'
filename = './celeba.zip'
if not os.path.exists(filename):
    print('Downloading CelebA dataset...')
    urllib.request.urlretrieve(url, filename)

if not os.path.exists('./data'):
    os.makedirs('./data')

print('Extracting CelebA dataset...')
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall('./data')
    

# celeba 데이터셋을 다운로드합니다.
trainset = torchvision.datasets.ImageFolder(root='./data/celeba/train', transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='./data/celeba/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

objects_strings = trainset.classes
print(objects_strings)

model = timm.create_model('resnet50', pretrained=True, num_classes=len(objects_strings))
model = model.to('cuda')
model = nn.DataParallel(model)

# 손실 함수와 최적화 알고리즘을 정의합니다.
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델을 학습합니다.
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # 100 미니배치마다 손실값을 출력합니다.
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    # 매 에폭마다 모델을 저장합니다.
    torch.save(model.state_dict(), 'model.pth')

print('Finished Training')

# 모델을 평가합니다.
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to('cuda')
        labels = labels.to('cuda')
        outputs = model(images)
        predicted = (outputs > 0.5).int()
        total += labels.size(0)
        correct += (predicted == labels).all(dim=1).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))