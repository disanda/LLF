import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import json
import yaml
from omegaconf import OmegaConf
import torch
from hydra.utils import instantiate, to_absolute_path
import numpy as np
import torchvision
from PIL import Image, ImageDraw, ImageFont
import math
import random
from pathlib import Path
from models.generators import Generator
from typing import List, Optional, Union
import logging
import torchvision.transforms as T
import tqdm
import utils.training_utils as training_utils
#from models.generators.stylegan2.stylegan2_wrap import StyleGAN2Generator # StyleGAN2v2
# sys.path.append("..")
# sns.set_context("poster")

# load and print config
save_imgs = True 
conf_path =  './utils/visual_config_ffhq.yaml'
cfg = yaml.load(open(conf_path), Loader=yaml.FullLoader)
cfg = OmegaConf.create(cfg)
dirs = cfg.n_dirs
#dirs = [i for i in range(85,100)]
#print(cfg)

use_gpu = True
device = torch.device("cuda" if use_gpu else "cpu")

# init models
model: torch.nn.Module = instantiate(cfg.model, k=cfg.k).to(device)
generator: torch.nn.Module = instantiate(cfg.generator, device=device, checkpoint_path=to_absolute_path(cfg.generator_path))#StyleGAN2Generator(device,truncation= 0.7,feature_layer = 'conv1', use_w = True, checkpoint_path = checkpoint_path)

# preload models
checkpoint = torch.load(to_absolute_path(cfg.model_path), map_location=device)
model.load_state_dict(checkpoint["model"])

# set to eval
model.eval()
generator.eval()

save_imgs = True
img_path = cfg.model_name + '_imgs'
if save_imgs and not os.path.exists(img_path):
    os.mkdir(img_path)

sign = lambda x: math.copysign(1, x)
class Visualizer:
    """Model evaluator
    Args:
        model: model to be evaluated
        generator: pretrained generator
        device: device on which to evaluate model
        n_samples: number of samples
    """

    def __init__(
        self,
        model: torch.nn.Module,
        generator: Generator,
        device: torch.device,
        n_samples: Union[int, str],
        n_dirs: Union[int, List[int]],
        alphas: List[int],
        iterative: bool = False,
        feed_layers: Optional[List[int]] = None,
        image_size: Optional[Union[int, List[int]]] = None,
        name: str = '',
        samples_random: bool = True,
        seed:int = 0,
    ) -> None:
        self.seed = seed
        # Logging
        self.logger = logging.getLogger()

        # Device
        self.device = device

        # Model
        self.name = name
        self.model = model
        self.generator = generator

        # Set to eval
        self.generator.eval()
        self.model.eval()

        # N Samples
        if isinstance(n_samples, int):
            if samples_random == True:
                self.samples = self.generator.sample_latent(n_samples)
            else:
                training_utils.set_seed(seed)
                self.samples = torch.randn(n_samples, 512).to(device)
                if self.generator.w_primary:
                    self.samples = self.generator.model.style(self.samples)
            self.samples = self.samples.to(self.device)
        else:
            print(f"Loading input file {n_samples}...")
            self.samples = torch.from_numpy(np.load(n_samples)).to(self.device) # [-1,512]
        print(self.samples.shape)

        #  Sub-sample Dirs
        if n_dirs == -1:
            self.dirs = list(range(self.model.k))
        elif isinstance(n_dirs, int):
            self.dirs = np.arange(0,n_dirs)
#             self.dirs = np.random.choice(self.model.k, n_dirs, replace=False)
        else:
            print(type(n_dirs))
            assert isinstance(n_dirs, list)
            self.dirs = n_dirs

        for i in self.dirs:
            if save_imgs and not os.path.exists(img_path+'/%d'%i):
                os.mkdir(img_path+'/%d'%i)

        # Alpha
        alphas = sorted(alphas)
        i = 0
        while alphas[i] < 0:
            i += 1
        self.neg_alphas = alphas[:i]

        if alphas[i] == 0:
            i += 1
        self.pos_alphas = alphas[i:]

        # Iterative
        self.iterative = iterative

        # Image Size
        if image_size:
            self.image_transform = T.Resize(image_size)
        else:
            self.image_transform = torch.nn.Identity()

        # Feed Layers
        self.feed_layers = feed_layers

    def visualize(self) -> float:
        """Generates images from the trained model

        Returns:
            (float) accuracy (on a 0 to 1 scale)

        """
        # Progress bar
        pbar = tqdm.tqdm(total=self.samples.shape[0], leave=False)
        pbar.set_description("Generating... ")

        # Set to eval
        self.generator.eval()
        self.model.eval()

        def _edit(z, alpha, ks):
            #  check if only one latent code is given
            assert z.shape[0] == 1 or z.shape[0] == len(
                ks
            ), """Only able to apply all directions to single latent code or
                apply each direction to single code"""
            self.model.alpha = alpha

            # Apply Directions
            zs = []
            for i, k in enumerate(ks):
                _i = i if z.shape[0] > 1 else 0
                zs.append(self.model.forward_single(z[_i : _i + 1, ...], k=k)) # [1,512] -> [512]
            zs = torch.cat(zs, dim=0) # [k,512]
            return zs

        # Helper function to generate images
        def _generate(zs, z=None):
            # Manipulate only asked layers
            if self.feed_layers is not None and z is not None:
                n_latent = self.generator.n_latent()

                zs_layers = []
                for i in range(n_latent):
                    if i in self.feed_layers:
                        zs_layers.append(zs)
                    else:
                        zs_layers.append(z.expand(zs.shape[0], -1))
                zs = zs_layers

            images = self.generator(zs).detach().cpu()
            return self.image_transform(images)

        # Loop
        with torch.no_grad():
            for i in range(self.samples.shape[0]):
                # Take a single sample
                z = self.samples[i : i + 1, ...]

                # Get original image
                orj_img = _generate(z) # [1, 3, 256, 256]

                if save_imgs:
                    for j in self.dirs:
                        img_path+'/%d'%j
                        torchvision.utils.save_image(orj_img*0.5+0.5, img_path+'/%d/'%j+"id-%d_alpha-0_seed-%d.png"%(i,self.seed), nrow=1)

                # Accumulator for images
                images = []

                #  Start with z and alpha = 0
                z_orig = z
                prev_alpha = 0
                for alpha in reversed(self.neg_alphas):
                    #  if iterative use last z and d(alpha)
                    _z = z if self.iterative else z_orig
                    _alpha = alpha - prev_alpha if self.iterative else alpha

                    z = _edit(_z, _alpha, ks=self.dirs)
                    one_neg_img = _generate(z, z_orig)
                    images.append(one_neg_img)
                    prev_alpha = alpha

                    if save_imgs:
                        for j in self.dirs:
                            img_path+'/%d'%j
                            single_img = one_neg_img[j]
                            torchvision.utils.save_image(single_img.view([1] + list(single_img.shape))*0.5+0.5, img_path+'/%d/'%j+"id-%d_neg-%d_seed-%d.png"%(i,alpha,self.seed), nrow=1)

                # Reverse images
                images = list(reversed(images))

                # Reset z and alpha
                z = z_orig
                prev_alpha = 0
                for alpha in self.pos_alphas:
                    #  if iterative use last z and d(alpha)
                    _z = z if self.iterative else z_orig
                    _alpha = alpha - prev_alpha if self.iterative else alpha

                    z = _edit(_z, _alpha, ks=self.dirs)
                    one_pos_img = _generate(z, z_orig)
                    images.append(one_pos_img)
                    prev_alpha = alpha

                    if save_imgs:
                        for j in self.dirs:
                            img_path+'/%d'%j
                            single_img = one_pos_img[j]
                            torchvision.utils.save_image(single_img.view([1] + list(single_img.shape))*0.5+0.5, img_path+'/%d/'%j+"id-%d_pos-%d_seed-%d.png"%(i,alpha,self.seed), nrow=1)

                #  Prepare final image
                images = torch.stack(images, dim=0)
                images = images.transpose(1, 0)
                col_orj_img = orj_img.repeat((images.size(0), 1, 1, 1))  # .unsqueeze(1)

                titles = []
                before_sign = -1
                imgs = []
                for ind, alpha in enumerate(self.neg_alphas + self.pos_alphas):
                    # append orijinal image
                    if sign(alpha) != before_sign:
                        imgs.append(col_orj_img)
                        titles.append("α=0")
                        before_sign = sign(alpha)

                    titles.append(f"α= {alpha:.3f}")
                    imgs.append(images[:, ind, ...])
                images = torch.stack(imgs).transpose(1, 0)*0.5+0.5

                images = images.reshape(
                    -1, images.size(-3), images.size(-2), images.size(-1)
                )

                imgs_grid = torchvision.utils.make_grid(
                    images,
                    nrow=len(self.neg_alphas) + len(self.pos_alphas) + 1,
                    padding=2,
                    pad_value=255,
                )

                fnt = ImageFont.truetype('./utils/FreeMono.ttf', 30)
#                 fnt = ImageFont.load_default(30)

                # get a drawing context
                img_alpha = Image.new("RGB", (imgs_grid.size(-1), 50), (255, 255, 255))
                d = ImageDraw.Draw(img_alpha)
                
                # draw alpha text
                for ind, text in enumerate(titles):
                    d.multiline_text(
                        (40 + ind * (images.size(-2) + 2), 10),
                        text,
                        font=fnt,
                        fill=(0, 0, 0),
                    )

                # get a drawing context
                img_k = Image.new(
                    "RGB", (100, imgs_grid.size(-2) + 50), (255, 255, 255)
                )
                d = ImageDraw.Draw(img_k)

                #  draw direction text
                for ind in range(len(self.dirs)):
                    d.multiline_text(
                        (10, 100 + ind * (images.size(-1) + 2)),
                        f"k={self.dirs[ind]}",
                        font=fnt,
                        fill=(0, 0, 0),
                    )

                img_alpha = T.ToTensor()(img_alpha)
                img_k = T.ToTensor()(img_k)

                print(img_k.shape)

                imgs_grid = torch.cat([img_alpha, imgs_grid], dim=-2)
                imgs_grid = torch.cat([img_k, imgs_grid], dim=-1)

                torchvision.utils.save_image(imgs_grid, f"%s_seed%d_%d.png"%(self.name, self.seed, i))

                # Update progress bar
                pbar.update()

        pbar.close()

visualizer = Visualizer(
    model=model,
    generator=generator,
    device=device,
    n_samples=cfg.n_samples,
    n_dirs=dirs,
    alphas=cfg.alphas,
    feed_layers=cfg.feed_layers,
    image_size=cfg.image_size,
    name = cfg.model_name,
    samples_random = cfg.sample_random,
    seed = cfg.seed
)

visualizer.visualize()

# baseline visual tool
# #  Helper function to edit latent codes
# def _edit(z, alpha, ks):
#     """
#         z: latent code to edit
#         alpha: magnitude of the edit
#         ks: directions to apply
#     """
#     #  check if only one latent code is given
#     assert z.shape[0] == 1 or z.shape[0] == len(
#         ks
#     ), "Only able to apply all directions to single latent code or apply each direction to single code"
#     model.alpha = alpha

#     # Apply Directions
#     zs = []
#     for i, k in enumerate(ks):
#         #_i = i if z.shape[0] > 1 else 0
#         zs.append(model.forward_single(z[i : i + 1, ...], k=k))
#     zs = torch.cat(zs, dim=0)
#     # print("################")
#     # print(model.sample_alpha())
#     # print("################")
#     return zs

# # Helper function to generate images
# def _generate(zs, z=None, feed_layers=None):
#     """
#         zs: z codes to feed into generator
#         z: original z code
#         feed_layers: targeted edit layers
#     """
#     # Manipulate only asked layers
#     if feed_layers is not None and z is not None:
#         n_latent = generator.n_latent()

#         zs_layers = []
#         for i in range(n_latent):
#             if i in feed_layers:
#                 zs_layers.append(zs)
#             else:
#                 zs_layers.append(z.expand(zs.shape[0], -1))
#         zs = zs_layers
#     # print(len(zs))
#     # if z != None: print(z.shape)
#     # print(feed_layers)
#     return generator(zs).detach().cpu()

# Visualizes images
# def visualize(
#     dir_ids,
#     feed_layers,
#     alphas=[-9,-6,-3,0,3,6,9],
#     feat_name=None,
#     seeds=[0],
#     iterative=False,
#     scale=5,
# ):
#     # process alphas
#     alphas = sorted(alphas)
#     i = 0
#     while alphas[i] < 0:
#         i += 1
#     neg_alphas = alphas[:i]

#     if alphas[i] == 0:
#         i += 1
#     pos_alphas = alphas[i:]
    
    
#     for seed in seeds:
#         # set seed
#         np.random.seed(seed)
        
#         # generate latent code
#         z = generator.sample_latent(1)
#         z = z.to(device)
    

#         with torch.no_grad():
#             # get original image
#             orj_img = _generate(z)

#             # images container
#             images = []

#             #  start with z and alpha = 0
#             z_orig = z
#             prev_alpha = 0
#             for alpha in reversed(neg_alphas):
#                 #  if iterative use last z and d(alpha)
#                 _z = z if iterative else z_orig
#                 _alpha = alpha - prev_alpha if iterative else alpha

#                 z = _edit(_z, _alpha, ks=dir_ids)
#                 images.append(_generate(z, z_orig, feed_layers=feed_layers))
#                 prev_alpha = alpha

#             # reverse images
#             images = list(reversed(images))

#             # reset z and alpha
#             z = z_orig
#             prev_alpha = 0
#             for alpha in pos_alphas:
#                 #  if iterative use last z and d(alpha)
#                 _z = z if iterative else z_orig
#                 _alpha = alpha - prev_alpha if iterative else alpha

#                 z = _edit(_z, _alpha, ks=dir_ids)
#                 images.append(_generate(z, z_orig,feed_layers=feed_layers))
#                 prev_alpha = alpha

#             #  prepare final image
#             images = torch.stack(images, dim=0)
#             images = images.transpose(1, 0)
#             col_orj_img = orj_img.repeat((images.size(0), 1, 1, 1))

#             titles = []
#             before_sign = -1
#             imgs = []
#             for ind, alpha in enumerate(neg_alphas + pos_alphas):
#                 # append orijinal image
#                 if sign(alpha) != before_sign:
#                     imgs.append(col_orj_img)
#                     titles.append("α=0")
#                     before_sign = sign(alpha)

#                 titles.append(f"α= {alpha:.3f}")
#                 imgs.append(images[:, ind, ...])
#             images = torch.stack(imgs).transpose(1, 0)
            
#             images = images.transpose(2,3)
#             images = images.transpose(3,4)

#             fig, axs = plt.subplots(
#                 nrows=images.shape[0],
#                 ncols=images.shape[1],
#                 figsize=(images.shape[1] * scale, images.shape[0] * scale))
#             axs = axs.reshape(images.shape[0], images.shape[1])

#             fig.suptitle(feat_name)
#             for i in range(images.shape[0]):
#                 axs[i][0].set_ylabel(f"k= {dir_ids[i]}")
#                 for j in range(images.shape[1]):
#                     axs[i][j].set_xlabel(titles[j])
#                     axs[i][j].set_xticks([])
#                     axs[i][j].set_yticks([])
#                     axs[i][j].imshow(images[i][j])
#     plt.show()

# change json file to try different models
# JSON_PATH = "./visualization/stylegan2_configs/stylegan2_ffhq.json"
# with open(JSON_PATH, "r") as f:
#     CONFIG_JSON = json.load(f)
# # DIRECTIONS = CONFIG_JSON["DIRECTIONS"]
# # LAYER_MAPS = CONFIG_JSON["LAYER_MAPS"]
# DIRECTIONS = CONFIG_JSON["DIRECTIONS_COMMON"]
# ALPHAS = CONFIG_JSON['ALPHAS']
# SEEDS = CONFIG_JSON['SEEDS']

# print("Annotated features:")
# feat_list = list(DIRECTIONS.keys())
# print(feat_list)

# feat_name = "unknow2"
# dir_ids = DIRECTIONS[feat_name][0]

# visualize(
#     dir_ids= [dir_ids],
#     feed_layers= DIRECTIONS[feat_name][1], # e.g., [10,11],  [6,7,8,9,10,11,12,13],
#     feat_name= feat_name, # ['hair_color' 2, 'hl_face', 'rotation', 'grin', 'hairstyle', 'bald', 'happy2serious', 'old', 'smile', 'lipstick'] 0:10
#     alphas= ALPHAS,
#     seeds= SEEDS
# )