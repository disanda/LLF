# titile

## pre-trained model

- https://github.com/rosinality/stylegan2-pytorch

- ./models/generators/stylegan2/stylegan2_pytorch/checkpoint/tf

## kmeans.py

>segmentation (local features)

segmentation cluster

truncation_alpha = 0.5 

<!-- #系数降低图像生成效果变差，但增加了样本多样性，保证分割效果 -->

## train.py

>learning layer wise & local features for direction model

- config file
./utils/conf.yaml

- segmentation file
./kmeans/16x16/kmeans_k_clusters_3_feature_size_16_samples_256_feature_layer_convs.3.pkl


## visual.py

> visualization of results

<!-- 可视化 -->

- config file

./utils/visual_config_ffhq.yaml

truncation_alpha = 0.85 


<!-- n_samples #样本个数 或 潜码文件 ./RFM_latentCLR/id1-i0-w600-norm301.878265-imgLoss2.196582.pt # 30 #./checkpoint/latent_code.npy  #id5_norm376.pt  #id2_norm375.pt #./checkpoint/latent_code.npy #./checkpoint/real_img/id10-i0-w1900-norm373.856293-imgLoss1.587639.pt


#系数提高图像效果变好，但降低了样本多样性


就效果而言：
层数越高，方向值需越小
层数越少，方向值需越大 -->


## pip list 

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

## baseline & previous works

- https://github.com/catlab-team/latentclr
- https://github.com/warmspringwinds/segmentation_in_style


<!-- 实验
    1.添加马 --已完成
      - 只做了两个类，新增一个类没做
      - 

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

文章修改
    1.图1加两个输入和输出
    2.删除MSE部分 -->