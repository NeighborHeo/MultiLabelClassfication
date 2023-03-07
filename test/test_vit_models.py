import os
import sys
pwd = os.path.realpath(__file__)
pwd = os.path.dirname(pwd)
parent = os.path.dirname(pwd)
print("pwd: ", pwd)
print("parent: ", parent)
sys.path.append(parent)

import models.vit_models as vit_models
import torch
import torch.nn as nn
from functools import partial

def main():
    print("main")
    model = vit_models.VisionTransformer(
        img_size=224, patch_size=16, num_heads=1, embed_dim=192, depth=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

if __name__ == "__main__":
    main()
