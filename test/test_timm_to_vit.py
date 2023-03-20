# pretrained vit model from timm
import timm
import torch
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
current_state_dict = model.state_dict()
# save the pretrained model
torch.save(current_state_dict, 'vit_tiny_patch16_224.pth')
model.eval()

# vit model from this repo
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.vit_models import VisionTransformer, vit_tiny_patch16_224
model2 = vit_tiny_patch16_224(pretrained=True)
weights = model.state_dict()
weights2 = model2.state_dict()


for k in weights.keys():
    same = torch.allclose(weights[k], weights2[k])
    if not same:
        # print(k, 'not same')
        shape_is_same = weights[k].shape == weights2[k].shape
        if shape_is_same:
            print(k, 'shape is same')
            # print(weights[k].shape)
            # print(weights2[k].shape)
            # print(weights[k])
            # print(weights2[k])
        else:
            print(k, 'shape is not same')
            # print(weights[k].shape)
            # print(weights2[k].shape)
            # print(weights[k])
            # print(weights2[k])
        # print(weights[k])
        # print(weights2[k])
    else:
        print(k, 'same')