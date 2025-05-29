import torch
import pickle
import legacy
import numpy as np
import torchvision
from models.generators import stylegan2_ada_networks

seed = 31
n_images = 1
np.random.seed(seed)

if torch.cuda.is_available():    
   device =  'cuda'
else:
   device = 'cpu'

model_path = './checkpoint/generators/stylegan2-ada-ffhq.pkl'
#model_path = '/Users/apple/Desktop/my-code/MyPaper-Code/LFL_code/LLF_local_model/checkpoint/generators/stylegan2-ada-ffhq.pkl'

# with open(model_path, 'rb') as f:
#     G = pickle.load(f)['G_ema']#.cuda()  # torch.nn.Module

G = legacy.load_model(model_path, device)

z = torch.randn([1, G.z_dim])#.cuda()    # latent codes
c = None  # class labels (not used in this example)

z1 = torch.from_numpy(np.random.RandomState(int(seed)).randn(n_images, G.z_dim)).to(device)
label = torch.zeros([n_images, G.c_dim], device=device)

#w = G.mapping(z, c, truncation_psi=0.7) # truncation_cutoff=8
w = G.get_ws(z1,label,truncation_psi=0.7)

#feats, img = G.synthesis(w, noise_mode='const', force_fp32=True) # force_fp32=True
#img = G(z, c)                           # NCHW, float32, dynamic range [-1, +1]

#mid_size= [-1, 4, 8, 16, 32, 64, ..., 1024]
feats, img = G.synthesis(w, mid_size=-1, noise_mode='const', force_fp32=True) # force_fp32=True

print(len(img))
print(feats.shape)
print(img.shape)

torchvision.utils.save_image(img*0.5+0.5, './stylegan_ada_seed=%d.png'%(seed), nrow=1)