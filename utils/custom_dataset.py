import torch
import numpy as np

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