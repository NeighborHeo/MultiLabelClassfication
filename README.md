# Train CIFAR10 with PyTorch

forked from https://github.com/kuangliu/pytorch-cifar.git

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
python main.py

# You can manually resume the training with: 
python main.py --resume --lr=0.01
```

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG19](https://arxiv.org/abs/1409.1556)              | 93.97%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 95.35%      |
| [GoogLeNet]                                           | 95.43%      |
| [DenseNet121]                                         | 95.52%      |
