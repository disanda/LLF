import torch
import joblib
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils.training_utils import *
from models.generators.stylegan2.stylegan2_wrap import StyleGAN2Generator # StyleGAN2v2
import torchvision
import time
import pickle

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

k_clusters = 5 # 4*4 : 3  8*8 : 5  16*16 : 7  32*32 : 9 
feature_size = 16 #pow(2,5)
img_size = 512
n = 16 #单词生成特征样本总数的倍数 (每次per_iter个样本)
per_iter =16
dataset='car' # horse, church, cat, car
feature_layer = 'convs.3'
truncation_alpha = 0.5 # 1， 0.7,  0.5
# kmean_save_path = './cat_%s_kmeans_k_clusters_%d_feature_size_%d_samples_%d_feature_layer_%s_trunc_%s.pkl'%(dataset,k_clusters,feature_size,n*per_iter,feature_layer,truncation_alpha)
# kmean_load_path = kmean_save_path
kmean_load_path = "./checkpoint/kmeans/car_kmeans_k_clusters_5_feature_size_16_samples_256_feature_layer_convs.3_trunc_0.5.pkl"
# n_samples_path = './checkpoint/latent_code.npy'

# #加载预训练模型
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
generator_path = './checkpoint/generators/stylegan2-car-config-f.pt'
generator = StyleGAN2Generator(device,truncation= truncation_alpha, feature_layer = feature_layer, use_w = False, checkpoint_path = generator_path, class_name=dataset)
generator.eval()

# ##生成feartures
features = None
imgs = None
for i in range(n):
  set_seed(i)
  sample_z = torch.randn(per_iter, 512).to(device) # use_w = False
  #sample_z = torch.from_numpy(np.load(n_samples_path)).to(device) # from numpy w [-1,512]
  #sample_z = torch.load(n_samples).to(self.device) # from pytorch real_image w [-1,18,512] 
  with torch.no_grad():
    out = generator.get_features(sample_z)
    img = generator(sample_z).cpu()
    #style = generator.model.style(z)
    #img_gen, out = generator.model([style], input_is_w=True, noise=generator.noise, randomize_noise = False, feature_layer_number_to_return=1, truncation=generator.truncation, truncation_latent=generator.latent_avg)

  one_input = img

  if imgs is None:
    imgs = one_input.cpu()
  else:
    aditional_imgs = one_input.cpu()
    imgs = torch.cat((imgs, aditional_imgs), axis=0)

  if features is None:
    features = out.cpu()
  else:
    additional_features = out.cpu()
    features = torch.cat((features, additional_features), axis=0)
  print("feats_finish_%d_th"%((i+1)*per_iter))

features_new = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]) #64*64*(n*10) , 512
arr = features_new.detach().cpu().numpy()

# #训练kmeans模型
# kmeans_model = KMeans(n_clusters=k_clusters, random_state=3, verbose=2)
# time_start=time.time()
# print('training start')
# kmeans_model.fit(arr)
# time_end=time.time()
# print('time cost: %s'%str(time_end-time_start))
# joblib.dump(kmeans_model,kmean_save_path)


## 加载kmeans模型
kmeans_model = joblib.load(kmean_load_path) #加载kmeans模型 or with open(kmean_load_path, "rb") as f: kmeans_model = pickle.load(f)
labels = kmeans_model.predict(arr) #  (64*64*(n*10), 512) => 64*64*(n*10)

# visual_model
centers = kmeans_model.cluster_centers_
labels_spatial =  kmeans_model.labels_.reshape(-1, feature_size, feature_size)  #64*64*(n*10)
#labels_spatial = labels.reshape(features.shape[0], features.shape[2], features.shape[3]) # [images, feature_size, feature_size]
print(labels_spatial.shape)
print(centers.shape)

## visual_predict
# io.imshow(norm_ip(imgs[img_id, :, :, :], min=-1.0, max=1.0).permute(1, 2, 0).numpy())
# plt.show()
for img_id in range(2): # features.shape[0]
  print('img_id_%d'%(img_id))
  #输出图像
  plt.axis('off') # 去坐标轴
  plt.xticks([]) # 去刻度
  plt.imshow(norm_ip(imgs[img_id, :, :, :], min=-1.0, max=1.0).permute(1, 2, 0).numpy())
  plt.savefig('img_id_%d.png'%(img_id),bbox_inches='tight',pad_inches = -0.1)
  plt.show()

  #输出segmentation, clusters
  plt.axis('off') # 去坐标轴
  plt.xticks([]) # 去刻度
  # plt.colorbar()
  plt.imshow(labels_spatial[img_id, :, :],cmap='rainbow') # cmap='rainbow' 'summer', io.show(), 没有colorbar
  plt.savefig('img_id_%d_clusters%d_alpha_%.2f.png'%(img_id,k_clusters,truncation_alpha),bbox_inches='tight',pad_inches = -0.1)
  plt.show()

  #输出每个cluster
  for i in range(centers.shape[0]):
    print('img_%d_%d_th_cluster'%(img_id,i))
    plt.axis('off') # 去坐标轴
    plt.xticks([]) # 去刻度
    plt.imshow( (labels_spatial[img_id, :, :]==i)*1. ,cmap='rainbow')
    plt.savefig('img_id_%d_cluster_%d_m1.png'%(img_id,i),bbox_inches='tight',pad_inches = -0.1)
    plt.show()

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