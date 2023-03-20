import unittest
import torch
import timm

if __name__ == '__main__':
    # model = timm.create_model('resnet50', pretrained=True, num_classes=10)
    # print(model)
    # mo
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=20)
    model = torch.nn.DataParallel(model)
    last_layer_name = list(model.module.named_children())[-1][0]
    params = [
        {'params': [p for n, p in model.module.named_parameters() if last_layer_name not in n], 'lr': 1e-4},
        {'params': [p for n, p in model.module.named_parameters() if last_layer_name in n], 'lr': 1e-3},
    ]
    