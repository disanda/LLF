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

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
generator_path = './checkpoint/generators/stylegan2-ffhq-config-f.pt'
generator = StyleGAN2Generator(device,truncation= 0.7,feature_layer = 'convs.6', use_w = False, checkpoint_path = generator_path)
print('#######################')

with torch.no_grad():
    set_seed(100000)
    w1 = torch.randn(10,512).to(device)
    x1 = generator(w1)
    #x = generator.partial_forward(w, "convs.5")
    set_seed(100001)
    w2 = torch.randn(10,512).to(device)
    set_seed(100002)
    w3 = torch.randn(10,512).to(device)

    x2 = generator(w2)

    f1 = generator.get_features(w1)
    f2 = generator.get_features(w2)
    f3 = generator.get_features(w3)

#x3 = torch.cat([x1,x2])
f1 = f1[:,:1,:,:]
f2 = f2[:,:1,:,:]
f3 = f3[:,:1,:,:]
print(f1.shape)
print(f2.shape)
print(f3.shape)

f4 = f2 - f1 
f5 = f3 - f1

#torchvision.utils.save_image(x3*0.5+0.5,'./face_10.jpg',nrow=10)

flag = 0 
for  a,b,c,d,e in zip(f1, f2, f3, f4, f5):
    torchvision.utils.save_image(a,'./f1_%d_convs9_64x64.png'%flag,nrow=1)
    torchvision.utils.save_image(b,'./f2_%d.png'%flag,nrow=1)
    torchvision.utils.save_image(c,'./f3_%d.png'%flag,nrow=1)
    torchvision.utils.save_image(d,'./f4_%d.png'%flag,nrow=1)
    torchvision.utils.save_image(e,'./f5_%d.png'%flag,nrow=1)
    flag = flag + 1

