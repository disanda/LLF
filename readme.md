
# LLF (Learning Layer-wise and Local Features in StyleGANs)

Official implementation and pre-trained models for "Towards Interpretable Face Morphing via Unsupervised Learning of Layer-wise and Local Features"

## Visualization 
<div align="center">
    <img src="./supplementary/Animation_demo_fps7_sclae025.gif", width="600">
</div>

## 1. Pre-trained model

### 1.1 Download StyleGAN model first

1. StyleGAN2-ada or StyleGAN2

  - Download "stylegan2-ffhq-config-f.pt" here:

https://drive.google.com/file/d/1QyDC3btIP4DUfiSmAkdJWSbKJsU60-Pv/view?usp=sharing

  - Download "stylegan2-ada-ffhq-config-f.pt" here:

https://github.com/NVlabs/stylegan2-ada-pytorch

2. Put the model to here: "./checkpoint/generators/"

3. Run test file to generate image

> python test_stylegan2.py

or 

> python test_stylegan2_ada.py

3. see details on: 

- https://github.com/rosinality/stylegan2-pytorch

- https://github.com/NVlabs/stylegan2-ada-pytorch


### 1.2 Download Direction Model

1. Download "most_recent_epoch6_best.pt" here  (as case 5 clusters on 32x32 features with layer depth 3): 

https://drive.google.com/file/d/1tI72PrrZTFf-GsjMF1UgGGVnKADNdLqa/view?usp=sharing

2. Save here "./checkpoint/direction_models/"

3. see './outputs/train.log' for details

## 2.Visualization_Code (visual.py)

> python visual.py  

### Note: Please set the Configure file carefully:

- visual_config_ffhq.yaml

1. where: "./utils/visual_config_ffhq.yaml"

2. generator_path: "./checkpoint/generators/stylegan2-ffhq-config-f.pt"

3. trained_model_path: "./outputs/Year-Month-Day/Hour-Min-Sec/most_recent_epoch6_best.pt"

4. truncation_alpha = 0.85 


### Here is different feature sizes before output image (N, 3,1024,1024):

- torch.Size([10, 512, 4, 4]), conv0 (const intputs) conv1 
- torch.Size([10, 512, 8, 8]) , convs.2, convs.3
- torch.Size([10, 512, 16, 16]) , convs.4, convs.5
- torch.Size([10, 512, 32, 32]) , convs.6, convs.7
- torch.Size([10, 512, 64, 64]) , convs.8, convs.9   # batch_size = 4, k = 28
- torch.Size([10, 256, 128, 128]) , convs.10, convs.11 # batch_size = 4, k = 20
- torch.Size([10, 128, 256, 256]) ,  convs.12, convs.13
- torch.Size([10, 64, 512, 512]) , convs.14, convs.15
- torch.Size([10, 32, 1024, 1024]) , convs.16, convs.17

and a guideline for morphing image as:

```
feed_layers:
# - 0  # hair [200,360]
# - 1 # hair-shape, face-shape  [80,240] (head-zoom) pose
# - 2 # eyeglasses, pose, hair-shape, face-shape, age, Gender
#- 3 # hair, face, identity, age,Gender, Mouth, Eyeglasses,   face-shape, xiaba, smile, pose, hair_more
#- 4 #  hair, mouth-shape face-shape [20,80] @@@ smile, mouth(related-expression or hair-man)  mouth-shape face-shape  bald

#- 5 # expression [feeling, mood] mouth-shape&color eye brow
- 6   #eye-shape, brow-shape 6-7  [20,40,80] face-color Beard-shape age
# - 7 # hair&eye [only man] face-color hair-color
# - 8 # color-light: [hair, face, background]

#- 9 # pupil
#- 10 # light [face] lips [red]
# - 11 # color lips(a few)
#- 12 # color details: lips, face, clothes
# - 13    # light, all img color
# - 14   # face color
# - 15  # hair color
#- 16 # weak color&light for [face, background]
# - 17 # background
```

<!-- 
n_samples #样本个数 或 潜码文件 ./RFM_latentCLR/id1-i0-w600-norm301.878265-imgLoss2.196582.pt # 30 #./checkpoint/latent_code.npy  #id5_norm376.pt  #id2_norm375.pt #./checkpoint/latent_code.npy #./checkpoint/real_img/id10-i0-w1900-norm373.856293-imgLoss1.587639.pt

#系数提高图像效果变好，但降低了样本多样性

就效果而言：
层数越高，方向值需越小
层数越少，方向值需越大 
-->

## 3. Learning Local features (kmeans.py)

### 3.1 Train

> python kmeans.py

- kmeans performed as segmentation model.
- segmentation clusters (feature sizes) within 3 (16x16), 5(16x16, 32x32), 7 (16x16, 32x32) , 9 (32x32, 64x64)
- truncation_alpha = 0.5 (better)
- results sava here: './results/segments/'

<!-- #系数降低图像生成效果变差，但增加了样本多样性，保证分割效果 -->

Save here: 

- kmean_save_path = './checkpoint/kmeans_segmentation/

### 3.2 Cases

- convs.5 = 32x32 feature size on StyleGAN2-FFHQ with clusters=9

>'./checkpoint/kmeans_segmentation/dataset_ffhq_kmeans_k_clusters_9_feature_size_32_samples_128_feature_layer_convs.5_trunc_0.5.pkl'

- convs.3 = 16x16 feautre size on StyleGAN2-FFHQ with clusters=5

>'./checkpoint/kmeans_segmentation/dataset_ffhq_kmeans_k_clusters_5_feature_size_16_samples_256_feature_layer_convs.3_trunc_0.5.pkl'

### 3.3 Visualization

>python kmeans_test.py


## 4. Learning direction model (train.py)

> python train.py

learning layer wise & local features via direction model

### 4.1 training config 

> ./utils/train_direction_model_conf.yaml

key hyper-paras:

- k: 30 (kearning 30 directions)

-  kmeans_model_path:  './checkpoint/kmeans/dataset_ffhq_kmeans_k_clusters_9_feature_size_32_samples_128_feature_layer_convs.5_trunc_0.5.pkl' #car_kmeans_k_clusters_5_feature_size_16_samples_256_feature_layer_convs.3_trunc_0.5.pkl'

- k_th_cluster: 5 

  > leraning the 5nd local features, here is eye features for the mentioned model

### 4.2 segmentation model 

> ./kmeans/16x16/kmeans_k_clusters_3_feature_size_16_samples_256_feature_layer_convs.3.pkl

- truncation_alpha = 0.7 or 0.85 (better) 

- output file

> ./output



## 5. Pip list 

```py
3090 / CUDA Version: 11.7 / windows 10 64:
Python 3.7.3

torch==1.9.1+cu111
torchvision==0.10.1+cu111
hydra-core==1.1.1
PyYAML==6.0
omegaconf==2.1.1
Pillow==9.1.1
pathlib2==2.3.4
joblib==1.2.0
scikit-image==0.15.0
scikit-learn==1.0.2
scipy==1.7.3


Previous:
Tesla V100-SXM2 / CUDA Version: 10.2 / ubuntu1~18.04.1
Python 3.8.3

torch==1.7.1
torchvision==0.8.2
Hydra==2.5
hydra-core==1.3.2
PyYAML==5.3.1
omegaconf==2.3.0
Pillow==7.2.0
scikit-image==0.16.2
scikit-learn==0.23.1
scipy==1.5.0
```

## 6. Reference

- https://github.com/catlab-team/latentclr
- https://github.com/warmspringwinds/segmentation_in_style

<!-- 
实验
    1.添加马 --已完成
      - 只做了两个类，新增一个类没做

    2.church
      - 再做一个类 2023/3/28（已完成）
    
    2.添加猫
      - cat_c0-5_k30_cf16_if32 1,6,18,27,29
      - cat_c1-5_k30_cf16_if32 
         layer2: 0-2,9,20, 13-23,29
      - 暂时完成
    
    3.完善车，多了一个类，每一层去看，效果更好，注意trunc 0.7 or 0.85
      - 只做了两个类，新增一个类没做
      - 再做一个类 2023/3/28
实验
-->
