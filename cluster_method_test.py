import torch
import joblib
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.training_utils import *
from models.generators.stylegan2.stylegan2_wrap import StyleGAN2Generator # StyleGAN2v2
from models.generators import stylegan2_ada_networks # StyleGAN2_ada
import torchvision
import time
import pickle
import legacy
import os

#z-> w, g_mapping
#w -> 0,7w,  truncation,
#torch.Size([10, 512, 4, 4]), conv1
#torch.Size([10, 512, 8, 8]) , convs.0, convs.1
#torch.Size([10, 512, 16, 16]) , convs.2, convs.3
#torch.Size([10, 512, 32, 32]) , convs.4, convs.5   
#torch.Size([10, 512, 64, 64]) , convs.6, convs.7   # batch_size = 4, k = 28
#torch.Size([10, 256, 128, 128]) , convs.8, convs.9 # batch_size = 4, k = 20
#torch.Size([10, 128, 256, 256]) , convs.10, convs.11
#torch.Size([10, 64, 512, 512]) , convs.12, convs.13
#torch.Size([10, 32, 1024, 1024]) , convs.14, convs.15

g_name = 'stylegan2' # 'stylegan2' or 'stylegan2-ada'
k_clusters = 9 # 4*4 : 3  8*8 : 5  16*16 : 7  32*32 : 9 
cluster_method = None
feature_size = 32 #pow(2,5), ada should feature_size*2
img_size = 1024 # 512, 1024
per_iter = 2 #单词迭代生成的特征样本，总样本数为 n * per_iter
dataset='ffhq' # ffhq, horse, church, cat, car
feature_layer = 'convs.5'
truncation_alpha = 0.5 # 1， 0.7,  0.5
# n_samples_path = './checkpoint/latent_code.npy'

### Visual_model, 加载kmeans模型
kmean_visual = False
if kmean_visual == True:
  cluster_method = 'km'
  kmean_sava_path = "./checkpoint/clustering_segmentation/dataset_ffhq_kmeans_k_clusters_9_feature_size_32_samples_8_feature_layer_convs.5_trunc_0.5.pkl"
  kmeans_model = joblib.load(kmean_sava_path) #加载kmeans模型 or with open(kmean_load_path, "rb") as f: kmeans_model = pickle.load(f)
  centers = kmeans_model.cluster_centers_
  labels_spatial =  kmeans_model.labels_.reshape(-1, feature_size, feature_size)  #64*64*(n*10)
  print(labels_spatial.shape)
  print(centers.shape)

## Visual_model, 加载gmm模型
gmm_visual = True # False
if gmm_visual == True:
  cluster_method = 'gmm'
  gmm_save_path = "./checkpoint/clustering_segmentation/dataset_ffhq_gmm_k_clusters_9_feature_size_32_samples_8_feature_layer_convs.5_trunc_0.5.pkl"
  gmm_model = joblib.load(gmm_save_path) 

# #加载预训练模型
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

#-------------stylegan2------------------
if g_name == 'stylegan2':
  generator_path = './checkpoint/generators/stylegan2-ffhq-config-f.pt'
  generator = StyleGAN2Generator(device,truncation= truncation_alpha, feature_layer = feature_layer, use_w = False, checkpoint_path = generator_path, class_name=dataset)

#-------------stylegan2-ada------------------
if g_name == 'stylegan2-ada':
  model_path = './checkpoint/generators/stylegan2-ada-ffhq.pkl'
  generator = legacy.load_model(model_path, device)

generator.eval()
seed = 33
set_seed(seed)
#sample_z = torch.randn(per_iter, 512).to(device) # use_w = False
sample_z = torch.from_numpy(np.random.RandomState(int(seed)).randn(per_iter, 512)).to(device)
#sample_z = torch.from_numpy(np.load(n_samples_path)).to(device) # from numpy w [-1,512]
#sample_z = torch.load(n_samples).to(self.device) # from pytorch real_image w [-1,18,512] 

with torch.no_grad():
    if g_name == 'stylegan2':
      out = generator.get_features_pre(sample_z.float()) # stylegan2
      imgs = generator(sample_z.float()).cpu()
    else:
      label = torch.zeros([1, generator.c_dim], device=device)
      w = generator.get_ws(sample_z,label,truncation_psi=0.7)
      out, imgs = generator.get_features(w) # stylegan2-ada
      out_feature_id = int(np.sqrt(feature_size) - 1)
      out = out[out_feature_id]

feats = out
feats_new = feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1]) # [B, C, H, W] => [B, C*H*W]
arr = feats_new.detach().cpu().numpy() #检查NAN、无穷大
arr[np.isnan(arr)]=0
arr[np.isinf(arr)]=0

if cluster_method == 'km':
  labels = kmeans_model.predict(arr)
elif cluster_method == 'gmm':
  labels = gmm_model.predict(arr)
labels_spatial_imgs = labels.reshape(feats.shape[0], feats.shape[2], feats.shape[3])

## visual_predict_2imgs
save_path = f'./outputs/segments/clusterM={cluster_method}_feature_layer={feature_layer}_k_clusters={k_clusters}/'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
for img_id in range(per_iter): # features.shape[0]
  print('img_id_%d'%(img_id))
  #输出图像
  plt.axis('off') # 去坐标轴
  plt.xticks([]) # 去刻度
  plt.imshow(norm_ip(imgs[img_id, :, :, :], min=-1.0, max=1.0).permute(1, 2, 0).cpu().numpy())
  plt.savefig(save_path+f'img_seed%d_id%d.png'%(seed,img_id),bbox_inches='tight',pad_inches = -0.1)
  plt.show()

  #输出segmentation, clusters
  plt.axis('off') # 去坐标轴
  plt.xticks([]) # 去刻度
  # plt.colorbar()
  plt.imshow(labels_spatial_imgs[img_id, :, :],cmap='rainbow') # cmap='rainbow' 'summer', io.show(), 没有colorbar
  plt.savefig(save_path+f'img_seed%d_id%d_clusters%d_alpha_%.2f.png'%(seed,img_id,k_clusters,truncation_alpha),bbox_inches='tight',pad_inches = -0.1)
  plt.show()

  #输出每个cluster
  for i in range(centers.shape[0]):
    print('img_%d_%d_th_cluster'%(img_id,i))
    plt.axis('off') # 去坐标轴
    plt.xticks([]) # 去刻度
    plt.imshow( (labels_spatial[img_id, :, :]==i)*1. ,cmap='rainbow')
    plt.savefig(save_path+f'img_id_%d_cluster_%d_m1.png'%(img_id,i),bbox_inches='tight',pad_inches = -0.1)
    plt.close()
    #plt.show()

    # plt.axis('off') # 去坐标轴
    # plt.xticks([]) # 去刻度
    # plt.imshow( ~(labels_spatial[img_id, :, :]==i)*1. ,cmap='rainbow')
    # plt.savefig('img_id_%d_cluster_%d_m2.png'%(img_id,i),bbox_inches='tight',pad_inches = -0.1)
    # plt.show()

#    io.imsave('./img_%d_%d_th_cluster.jpg'%(img_id,i), labels_spatial[img_id, :, :], cmap='rainbow')
    # io.imsave('./img_%d_%d_th_cluster_m1.jpg'%(img_id,i),  (labels_spatial[img_id, :, :]==i)*1. ,cmap='rainbow')
    # io.imsave('./img_%d_%d_th_cluster_m2.jpg'%(img_id,i), ~(labels_spatial[img_id, :, :]==i)*1. ,cmap='rainbow')
    # io.imshow(labels_spatial[img_id+1, :, :],cmap='rainbow')
    # plt.show()
    # io.imshow((labels_spatial[img_id+1, :, :]==i)*1. ,cmap='rainbow')
    # plt.show()
    # io.imshow(labels_spatial[img_id+2, :, :],cmap='rainbow')
    # plt.show()
    # io.imshow((labels_spatial[img_id+2, :, :]==i)*1. ,cmap='rainbow')
    # plt.show()
    # io.imshow( ( (labels_spatial[img_id, :, :]==0) | (labels_spatial[img_id, :, :]==1) ).astype(int),cmap='rainbow')
    # plt.show()

#torchvision.utils.save_image(imgs*0.5+0.5,'k-means.jpg')

# if img.shape[3] != img_size:
#   one_input = torch.nn.functional.interpolate(img,size=(img_size, img_size),mode='bilinear',align_corners=True).clamp(min=-1.0, max=1.0).detach()
# else:
#   one_input = img