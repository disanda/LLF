import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import time
from typing import List, Optional
import torch
import tqdm
from pathlib import Path
import utils.training_utils as training_utils
import utils.metrics as metrics
import utils.log_utils as log_utils
from models.generators.stylegan2.stylegan2_wrap import StyleGAN2Generator # StyleGAN2v2
import torchvision
from utils.training_utils import *

# ------------------ test  styleGAN2-ffhq ---------------
# - torch.Size([10, 512, 4, 4]), conv0 (const intputs) conv1 
# - torch.Size([10, 512, 8, 8]) , convs.0, convs.1
# - torch.Size([10, 512, 16, 16]) , convs.2, convs.3
# - torch.Size([10, 512, 32, 32]) , convs.4, convs.5
# - torch.Size([10, 512, 64, 64]) , convs.6, convs.7   # batch_size = 4, k = 28
# - torch.Size([10, 256, 128, 128]) , convs.8, convs.9 # batch_size = 4, k = 20
# - torch.Size([10, 128, 256, 256]) ,  convs.10, convs.11
# - torch.Size([10, 64, 512, 512]) , convs.12, convs.13
# - torch.Size([10, 32, 1024, 1024]) , convs.14, convs.15

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
generator_path = './checkpoint/generators/stylegan2-ffhq-config-f.pt'
generator = StyleGAN2Generator(device,truncation= 1 ,feature_layer = 'convs.14', use_w = True, checkpoint_path = generator_path)
print('#######################')
#print(generator)

with torch.no_grad():
    set_seed(77249920)
    z1 = torch.randn(1,512).to(device)
    label = torch.zeros([1, 0], device=device) # G.c_dim = 0
    ws_original = generator.get_ws(z1,label,truncation_psi=0.7)
    handle_size = 128
    feature_original, img_mid_original = generator.get_features(ws_original,x=None, img=None, mid_size= handle_size)
    img = generator(ws_original)
    
    z2 = torch.from_numpy(np.random.RandomState(int(3)).randn(1, 512)).to(device) # G.z_dim = 512
    label = torch.zeros([1, 0], device=device) # G.c_dim = 0
    ws_original2 = generator.get_ws(z2,label,truncation_psi=0.7)
    _, img_show_original2 = generator.get_features(ws=ws_original,x=feature_original,img=img_mid_original,mid_size=handle_size)

print(feature_original.shape)
print(img_mid_original.shape)
print(img_mid_original.mean())
#print(img_show_original.shape)
torchvision.utils.save_image(img_mid_original*0.5+0.5, './img_mid_original_mean.png')
torchvision.utils.save_image(img_mid_original, './img_mid_original.png')
torchvision.utils.save_image(img*0.5+0.5, './img2.png')
torchvision.utils.save_image(img_show_original2*0.5+0.5, './img_show_original2.png')
