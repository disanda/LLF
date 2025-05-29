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

g_name = 'stylegan2' # 'stylegan2' or 'stylegan2-ada'
k_clusters = 9 # 4*4 : 3  8*8 : 5  16*16 : 7  32*32 : 9 
feature_size = 32 #pow(2,5) img_size = 512 or 1024
n = 1 # iteration for generating 样本生成的迭代次数
per_iter = 100 # 单词迭代生成的特征样本，总样本数为 n * per_iter
dataset='ffhq' # ffhq, horse, church, cat, car
feature_layer = 'convs.5'
truncation_alpha = 0.5 # 1， 0.7,  0.5
kmean_save_path = './checkpoint/kmeans_segmentation/dataset_%s_kmeans_k_clusters_%d_feature_size_%d_samples_%d_feature_layer_%s_trunc_%s.pkl'%(dataset,k_clusters,feature_size,n*per_iter,feature_layer,truncation_alpha)
# n_samples_path = './checkpoint/latent_code.npy'

# loading stylegan generator 加载预训练模型
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
generator_path = './checkpoint/generators/stylegan2-ffhq-config-f.pt'
generator = StyleGAN2Generator(device,truncation= truncation_alpha, feature_layer = feature_layer, use_w = False, checkpoint_path = generator_path, class_name=dataset)
generator.eval()

# ------------ generate feartures (different sizes)--------------
features = None
imgs = None
for i in range(n):
  set_seed(i)
  sample_z = torch.randn(per_iter, 512).to(device) # use_w = False
  #sample_z = torch.from_numpy(np.load(n_samples_path)).to(device) # from numpy w [-1,512]
  #sample_z = torch.load(n_samples).to(self.device) # from pytorch real_image w [-1,18,512] 
  with torch.no_grad():
    if g_name == 'stylegan2':
      out = generator.get_features(sample_z) # stylegan2
      img = generator(sample_z).cpu()
    else:
      label = torch.zeros([1, generator.c_dim], device=device)
      w = generator.get_ws(sample_z,label,truncation_psi=0.7)
      out, img = generator.get_features(w) # stylegan2-ada
      out_feature_id = int(np.sqrt(feature_size) - 1)
      out = out[out_feature_id]

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
    features = torch.cat((features, additional_features), axis=0) # [bs, 512, 16, 16] + [bs, 512, 16, 16] => [32,512,16,16]
  print("features_finish_at_iter_%d"%((i+1)))
  print(features.shape)

features_new = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]) #[bs, 512, 16, 16] -> [bs, 16, 16, 512] -> [bs, 512x16x16]
arr = features_new.detach().cpu().numpy()

#训练kmeans模型
kmeans_model = KMeans(n_clusters=k_clusters, random_state=3, verbose=2)
time_start=time.time()
print('training start')
kmeans_model.fit(arr)
time_end=time.time()
print('time cost: %s'%str(time_end-time_start))
joblib.dump(kmeans_model,kmean_save_path)


