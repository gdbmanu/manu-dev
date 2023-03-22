#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from LogGabor import LogGabor
from what_where.main import init, MNIST
from what_where.where import RetinaFill, WhereShift, RetinaBackground, RetinaMask
from POLO.PYramid2 import cropped_pyramid, local_filter, get_K, log_gabor_transform
from POLO.PYramid2 import inverse_pyramid, get_K_inv, inverse_gabor
from utils import view_data
from typing import List, Tuple

import pickle

width = {'in': 32, 'out': 64, 'ext': 128}

n_levels = {'in': 3, 'out': 3, 'ext': 3} 

base_levels = 2

n_color = {'in': 3, 'out': 3, 'ext': 3}

color = True

color_mode= 'rgb' #'lab' # 'hsv' #True

r_min = {'in': width['in'] / 4, 'out': width['out'] / 4, 'ext': width['ext'] / 4}

r_max = {'in': width['in'] / 2, 'out': width['out'] / 2, 'ext': width['ext'] / 2}

n_sublevel = n_eccentricity = {'in': 2, 'out': 4, 'ext': 8}

n_azimuth = {'in': 16, 'out': 32, 'ext': 64}

n_theta = {'in': 8, 'out': 8, 'ext': 8}

n_phase = {'in': 1, 'out': 1, 'ext': 1}

do_mask = False

gauss = False



lg={}
for k in ['in', 'out', 'ext']:
    pe = {'N_X': width[k], 'N_Y': width[k], 'do_mask': do_mask, 'base_levels':
              base_levels, 'n_theta': 0, 'B_sf': np.inf, 'B_theta': np.inf ,
          'use_cache': True, 'figpath': 'results', 'edgefigpath':
              'results/edges', 'matpath': 'cache_dir', 'edgematpath':
              'cache_dir/edges', 'datapath': 'database/', 'ext': '.pdf', 'figsize':
              14.0, 'formats': ['pdf', 'png', 'jpg'], 'dpi': 450, 'verbose': 0}   

    lg[k] = LogGabor(pe)



K = {}
for k in ['in', 'out', 'ext']:
    K[k] = get_K(width=width[k],
          n_sublevel = n_sublevel[k], 
          n_azimuth = n_azimuth[k], 
          n_theta = n_theta[k],
          n_phase = n_phase[k], 
          r_min = r_min[k], 
          r_max = r_max[k], 
          log_density_ratio = 2, 
          verbose=True,
          lg=lg[k])




args = init(filename='2020-07-01')
args


# In[7]:


class DualCroppedPyramid(object):
    def __init__(self, width, 
                 base_levels, 
                 color=color, 
                 do_mask=do_mask, 
                 verbose=False, 
                 n_levels=None, 
                 color_mode='rgb'):
        self.width = width
        self.base_levels = base_levels
        self.color = color
        self.do_mask = do_mask
        self.verbose = verbose
        self.n_levels = n_levels
        self.color_mode = color_mode

    def __call__(self, img):
        img_crop = {}
        for k in ['in', 'out', 'ext']:
            img_crop_part, level_size = cropped_pyramid(img.unsqueeze(0), 
                                               width=self.width[k], 
                                               base_levels=self.base_levels,
                                               color=self.color, 
                                               do_mask=self.do_mask, 
                                               verbose=self.verbose,
                                               squeeze=True,
                                               gauss=gauss,
                                               n_levels=self.n_levels[k],
                                               color_mode=self.color_mode)
            #print(img_crop_part.shape)
            img_crop[k] = img_crop_part[:self.n_levels[k]-1,...]
        return img, img_crop   




class DualLogGaborTransform(object):
    def __init__(self, K=K, color=color, verbose=False):
        self.K = K
        self.color = color
        self.verbose = verbose

    def __call__(self, cropped_pyr):
        img = cropped_pyr[0]
        img_crop = cropped_pyr[1]
        log_gabor_coeffs = {}
        for k in ['in', 'out', 'ext']:
            log_gabor_coeffs[k] = log_gabor_transform(img_crop[k].unsqueeze(1), K[k], color=self.color).squeeze(1)
        
        return img, log_gabor_coeffs




class DualLogGaborReshape(object):
    def __init__(self, n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase, color=color, verbose=False):
        self.n_levels = n_levels
        self.n_color = n_color
        self.n_eccentricity = n_eccentricity 
        self.n_azimuth = n_azimuth 
        self.n_theta = n_theta
        self.n_phase = n_phase  
        self.color = color
        self.verbose = verbose

    def __call__(self, log_gabor_transform):
        img = log_gabor_transform[0]
        log_gabor_coeffs = log_gabor_transform[1]
        for k in ['in', 'out', 'ext']:
            # n_batch, n_levels, n_color, n_sublevel, n_azimuth, n_theta, n_phase
            # x      , 0       , 1      , 2         , 3        , 4      , 5
            # 
            #print(log_gabor_coeffs[k].shape)
            log_gabor_coeffs[k] = log_gabor_coeffs[k].permute(1, 4, 5, 0, 2, 3).contiguous()
            log_gabor_coeffs[k] = log_gabor_coeffs[k].view(self.n_color[k]*self.n_theta[k]*self.n_phase[k], 
                                                     (self.n_levels[k]-1) * self.n_eccentricity[k], 
                                                     self.n_azimuth[k])
 
        return img, log_gabor_coeffs



v_mean = np.array([0.485, 0.456, 0.406])
v_std = np.array([0.229, 0.224, 0.225])
transforms_norm = transforms.Normalize(mean=v_mean, std=v_std) # to normalize colors on the imagenet dataset

image_size = 256

polo_transform =  transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            #transforms.AutoAugment(), # https://pytorch.org/vision/master/transforms.html#torchvision.transforms.AutoAugment
            transforms.ToTensor(),      # Convert the image to pyTorch Tensor data type.
            transforms_norm, 
            DualCroppedPyramid(width=width, 
                  base_levels=base_levels,
                  color=color,
                  n_levels=n_levels),
            DualLogGaborTransform(K=K, color=color),
            DualLogGaborReshape(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase, color=color),
        ])

# In[17]:


#image_path = "/envau/work/brainets/dauce.e/data/animal/"
image_path = "/media/manu/Seagate Expansion Drive/Data/animal/"
#image_path = "/run/user/1001/gvfs/sftp:host=bag-008-de03/envau/work/brainets/dauce.e/data/animal/"
#image_path = "../data/animal/"

image_dataset = { 'train' : datasets.ImageFolder(
                            image_path+'train', 
                            transform=polo_transform
                        ),
                  'test' : datasets.ImageFolder(
                            image_path+'test', 
                            transform=polo_transform
                        )
                }

# In[18]:


dataset_size = {'train' : len(image_dataset['train']),
                'test' : len(image_dataset['test'])}

dataset_size['train'], dataset_size['test']


# In[19]:


args.batch_size = 50
num_workers = 1

dataloader = { 'train' : torch.utils.data.DataLoader(
                            image_dataset['train'], batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=num_workers,
                        ),
               'test' : torch.utils.data.DataLoader(
                            image_dataset['test'], batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=num_workers,
                        )
             }



# # Creating an Attention Transformer model with log-polar entry (POLO-STN)

# In[28]:


transform_in =  transforms.Compose([
            DualCroppedPyramid(width=width, 
                  base_levels=base_levels,
                  color=color,
                  n_levels=n_levels),
            DualLogGaborTransform(K=K, color=color),
            DualLogGaborReshape(n_levels, n_color, n_eccentricity, n_azimuth, n_theta, n_phase, color=color),
        ])


# In[29]:


def negentropy_loss(model, z):
    z_mean = torch.mean(z, dim=0) + 1e-6
    z_std = torch.std(z, dim=0)
    if model.do_stn or args.radius > 0:
        p = torch.distributions.Normal(torch.ones_like(z)*z_mean, torch.ones_like(z) * z_std)
    else:
        p = torch.distributions.Normal(torch.zeros_like(z), 1e-6 * torch.ones_like(z))

    #p = torch.distributions.Normal(torch.ones_like(z)*z_mean, torch.ones_like(z) * z_std)
    return p.log_prob(z).sum()

def kl_divergence(model, z):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    if args.radius > 0:
        p = torch.distributions.Normal(torch.zeros_like(z), args.radius * torch.ones_like(z))
    else:
        p = torch.distributions.Normal(torch.zeros_like(z), 1e-6 * torch.ones_like(z))


    # 2. get the probabilities from the equation
    #log_qzx = model.q.log_prob(z)
    log_pz = p.log_prob(z)

    z_mean = torch.mean(z, dim=0)
    z_std = torch.std(z, dim=0) + 1e-6
    #print(z_std)
    q = torch.distributions.Normal(torch.ones_like(z)*z_mean, torch.ones_like(z) * z_std)
    log_qzx = q.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    
    # sum over last dim to go from single dim distribution to multi-dim
    kl = model.LAMBDA * kl.sum()
    return kl

class Polo_AttentionTransNet(nn.Module):
    
    def __init__(self, do_stn=True, do_what=False, deterministic=False, LAMBDA=.1):
        super(Polo_AttentionTransNet, self).__init__()
        
        self.do_stn = do_stn
        self.do_what = do_what
        self.deterministic = deterministic
        
        self.LAMBDA = LAMBDA

        ##  The what pathway
        self.wloc0 = nn.Conv2d(n_color['ext'] * n_theta['ext'] * n_phase['ext'], 
                              50, 3, padding=1)
        self.wloc1a = nn.Conv2d(50, 100, 3, padding=1)
        self.wloc1b = nn.Conv2d(n_color['out'] * n_theta['out'] * n_phase['out'], 
                              100, 3, padding=1)
        self.wloc2a = nn.Conv2d(100, 200, 3, padding=1)
        self.wloc2b = nn.Conv2d(100, 200, 3, padding=1)
        self.wloc2c = nn.Conv2d(n_color['in'] * n_theta['in'] * n_phase['in'], 
                               200, 3, padding=1)
        self.wloc3 = nn.Conv2d(200, 500, 3, padding=1)
        self.wloc4 = nn.Conv2d(500, 1000, 3, padding=1)
        self.wloc5_short = nn.Linear(1000 * (((n_levels['in']-1) * n_eccentricity['in'] * 3) // 8 * n_azimuth['in'] // 8), 2, bias=False)
        self.wloc5 = nn.Linear(1000 * (((n_levels['in']-1) * n_eccentricity['in'] * 3) // 8 * n_azimuth['in'] // 8), 1000)
        self.wloc6 = nn.Linear(1000, 2, bias=False)

        ##  The where pathway        
        self.loc0 = nn.Conv2d(n_color['ext'] * n_theta['ext'] * n_phase['ext'], 
                              50, 5, padding=2, stride=2)
        self.loc1a = nn.Conv2d(50, 100, 5, padding=2, stride=2)
        self.loc1b = nn.Conv2d(n_color['out'] * n_theta['out'] * n_phase['out'], 
                              50, 5, padding=2, stride=2)
        self.loc2a = nn.Conv2d(100, 200, 5, padding=2, stride=2)
        self.loc2b = nn.Conv2d(50, 100, 5, padding=2,stride=2)
        self.loc2c = nn.Conv2d(n_color['in'] * n_theta['in'] * n_phase['in'], 50, 5, padding=2,stride=2)
        self.loc3 = nn.Linear((n_levels['in']-1) * n_eccentricity['in'] // 2 * n_azimuth['in'] // 2 * (50+100+200), 1000)
        self.loc4 = nn.Linear(1000, 1000)
        self.mu = nn.Linear(1000, 2, bias=False)
        self.logvar = nn.Linear(1000, 2, bias=False)
        

# In[30]:
class Polo_ChanAttentionTransNet(nn.Module):
    
    def __init__(self, do_stn=True, do_what=False, deterministic=False, LAMBDA=.1):
        super(Polo_ChanAttentionTransNet, self).__init__()
        
        self.do_stn = do_stn
        self.do_what = do_what
        self.deterministic = deterministic
        
        self.LAMBDA = LAMBDA

        ##  The what pathway
        self.wloc0 = nn.Conv2d(n_color['ext'] * n_theta['ext'] * n_phase['ext'], 
                              50, 3, padding=1)
        self.wloc1a = nn.Conv2d(50, 100, 3, padding=1)
        self.wloc1b = nn.Conv2d(n_color['out'] * n_theta['out'] * n_phase['out'], 
                              100, 3, padding=1)
        self.wloc2a = nn.Conv2d(100, 200, 3, padding=1)
        self.wloc2b = nn.Conv2d(100, 200, 3, padding=1)
        self.wloc2c = nn.Conv2d(n_color['in'] * n_theta['in'] * n_phase['in'], 
                               200, 3, padding=1)
        self.wloc3 = nn.Conv2d(200, 500, 3, padding=1)
        self.wloc4 = nn.Conv2d(500, 1000, 3, padding=1)
        self.wloc5_short = nn.Linear(1000 * (((n_levels['in']-1) * n_eccentricity['in'] * 3) // 8 * n_azimuth['in'] // 8), 2, bias=False)
        self.wloc5 = nn.Linear(1000 * (((n_levels['in']-1) * n_eccentricity['in'] * 3) // 8 * n_azimuth['in'] // 8), 1000)
        self.wloc6 = nn.Linear(1000, 2, bias=False)

                
        ##  The where pathway        
        self.loc_ext_1 = nn.Conv2d(n_color['ext'] * n_theta['ext'] * n_phase['ext'], 
                              128, 3, padding=1, stride=1)
        self.loc_ext_2 = nn.Conv2d(32, 128, 5, padding=2, stride=1)
        self.loc_ext_3 = nn.Conv2d(32, 128, 7, padding=3, stride=2)

        self.loc_out_1 = nn.Conv2d(n_color['out'] * n_theta['out'] * n_phase['out'], 
                              128, 3, padding=1, stride=1)
        self.loc_out_2 = nn.Conv2d(32, 128, 5, padding=2,stride=1)

        self.loc_in_1 = nn.Conv2d(n_color['in'] * n_theta['in'] * n_phase['in'], 
                              128, 3, padding=1, stride=1)
        self.loc_in_2 = nn.Conv2d(32, 128, 5, padding=2,stride=1)

        #self.upsample_in = nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2)       
        self.loc_in_out = nn.Conv2d(32, 128, 7, padding=3,stride=1)

        #self.upsample_out = nn.ConvTranspose2d(32, 32, 3, padding=1, stride=2)   
        self.loc_all = nn.Conv2d(32, 128, 9, padding=4,stride=2)

        final_dim = 3 * (n_levels['out']-1) * n_eccentricity['out']//2 * n_azimuth['out']//2
        self.fc_where = nn.Linear(final_dim * 32, final_dim)
        self.mu = nn.Linear(final_dim, 2, bias=False)
        self.logvar = nn.Linear(final_dim, 2, bias=False)
        
        #self.loc4.weight.data.zero_()
        #self.loc4.bias.data.zero_()

        self.downscale = nn.Parameter(torch.tensor([[1, 0], [0, 1]],
                                                   dtype=torch.float),
                                      requires_grad=False)

    def stn(self: object, x: torch.Tensor, x_polo: torch.Tensor) -> Tuple[torch.Tensor]:
    
        if self.do_stn:
            xs_part = {}


            c_in = 128
            c_out = 32

            # IN
            w_in = (n_levels['in']-1) * n_eccentricity['in']
            h_in = n_azimuth['in']

            x_in = F.relu(self.loc_in_1(x_polo['in']))
            x_in = x_in.view(-1, c_in, w_in * h_in).permute(0, 2, 1)
            x_in = nn.MaxPool1d(4)(x_in)
            x_in = x_in.permute(0,2,1).view(-1, c_out, w_in, h_in)

            x_in = F.relu(self.loc_in_2(x_in))
            x_in = x_in.view(-1, c_in, w_in * h_in).permute(0, 2, 1)
            x_in = nn.MaxPool1d(4)(x_in)
            x_in = x_in.permute(0,2,1).view(-1, c_out, w_in, h_in)

            x_in = nn.Upsample(scale_factor=2, mode='nearest')(x_in)

            # OUT
            w_out = (n_levels['out']-1) * n_eccentricity['out']
            h_out = n_azimuth['out']

            x_out = F.relu(self.loc_out_1(x_polo['out']))
            x_out = x_out.view(-1, c_in, w_out * h_out).permute(0, 2, 1)
            x_out = nn.MaxPool1d(4)(x_out)
            x_out = x_out.permute(0,2,1).view(-1, c_out, w_out, h_out)

            x_out = F.relu(self.loc_out_2(x_out))
            x_out = x_out.view(-1, c_in, w_out * h_out).permute(0, 2, 1)
            x_out = nn.MaxPool1d(4)(x_out)
            x_out = x_out.permute(0,2,1).view(-1, c_out, w_out, h_out)

            x_in_out = torch.cat((x_in, x_out), dim=2)

            x_in_out = F.relu(self.loc_in_out(x_in_out))
            x_in_out = x_in_out.view(-1, c_in, 2 * w_out * h_out).permute(0, 2, 1)
            x_in_out = nn.MaxPool1d(4)(x_in_out)
            x_in_out = x_in_out.permute(0,2,1).view(-1, c_out, 2 * w_out, h_out)

            # EXT
            w_ext = (n_levels['ext']-1) * n_eccentricity['ext']
            h_ext = n_azimuth['ext']

            x_ext = F.relu(self.loc_ext_1(x_polo['ext']))
            x_ext = x_ext.view(-1, c_in, w_ext * h_ext).permute(0, 2, 1)
            x_ext = nn.MaxPool1d(4)(x_ext)
            x_ext = x_ext.permute(0,2,1).view(-1, c_out, w_ext, h_ext)

            x_ext = F.relu(self.loc_ext_2(x_ext))
            x_ext = x_ext.view(-1, c_in, w_ext * h_ext).permute(0, 2, 1)
            x_ext = nn.MaxPool1d(4)(x_ext)
            x_ext = x_ext.permute(0,2,1).view(-1, c_out, w_ext, h_ext)

            x_ext = F.relu(self.loc_ext_3(x_ext)) # stride=2
            x_ext = x_ext.view(-1, c_in, w_ext//2 * h_ext//2).permute(0, 2, 1)
            x_ext = nn.MaxPool1d(4)(x_ext)
            x_ext = x_ext.permute(0,2,1).view(-1, c_out, w_ext//2, h_ext//2)

            x_all = torch.cat((x_in_out, x_ext), dim=2)

            x_all = F.relu(self.loc_all(x_all))
            x_all = x_all.view(-1, c_in, 3 * w_out//2 * h_out//2).permute(0, 2, 1)
            x_all = nn.MaxPool1d(4)(x_all)
            x_all = x_all.permute(0,2,1).view(-1, c_out * 3 * w_out//2 * h_out//2)

            x_all = F.relu(self.fc_where(x_all))
            
            mu = self.mu(x_all)
            if self.deterministic:
                sigma = args.radius * torch.ones_like(mu)  
                self.q = torch.distributions.Normal(mu, sigma)  
                z = mu
            else:
                logvar = self.logvar(x_all) + 5
                sigma = torch.exp(-logvar / 2)
                self.q = torch.distributions.Normal(mu, sigma)      
                z = self.q.rsample()
            
            print(z[0,...])
            theta = torch.cat((self.downscale.unsqueeze(0).repeat(
                                z.size(0), 1, 1), z.unsqueeze(2)),
                                  dim=2)
        
            grid_size = torch.Size([x.size()[0], x.size()[1], 256, 256])
            grid = F.affine_grid(theta, grid_size)
            x = F.grid_sample(x, grid)

        else:
            mu = torch.tensor([0, 0],dtype=torch.double)
            mu = mu.unsqueeze(0).repeat(x.size()[0], 1)   
            sigma = torch.tensor([1, 1],dtype=torch.double)
            sigma = sigma.unsqueeze(0).repeat(x.size()[0], 1)    
            
            if self.do_what and args.radius > 0:
                self.q = torch.distributions.Normal(mu, args.radius*sigma)
                z = self.q.rsample().to(device)
                print(z[0,...])
                theta = torch.cat((self.downscale.unsqueeze(0).repeat(
                                z.size(0), 1, 1), z.unsqueeze(2)),
                                  dim=2)
        
                grid_size = torch.Size([x.size()[0], x.size()[1], 256, 256])
                grid = F.affine_grid(theta, grid_size)
                x = F.grid_sample(x, grid)
            else:
                z = torch.tensor([0, 0],dtype=torch.float)
                z = z.unsqueeze(0).repeat(x.size()[0], 1).to(device)

                theta = nn.Parameter(torch.tensor([[1, 0, 0], [0, 1, 0]],
                                                    dtype=torch.float),
                                        requires_grad=False)
                theta = theta.unsqueeze(0).repeat(x.size()[0], 1, 1)
            
              
        return x, theta, z

    def forward(self, x, x_polo):
        # transform the input
        x, theta, z = self.stn(x, x_polo)
        
        if self.do_stn or (self.do_what and args.radius > 0):
        
            w_x_polo ={'in': torch.zeros_like(x_polo['in']),
                       'out': torch.zeros_like(x_polo['out']),
                       'ext': torch.zeros_like(x_polo['ext'])}
            for i in range(args.batch_size):
                d, w = transform_in(x[i,...])
                w_x_polo['in'][i,...] = w['in']
                w_x_polo['out'][i,...] = w['out']
                w_x_polo['ext'][i,...] = w['ext']
        else:
            w_x_polo = x_polo
                                           

        # print(x.shape)
        # Perform the usual forward pass

        ya = F.relu(self.wloc0(w_x_polo['ext']))
        ya = nn.MaxPool2d(2)(ya)     
        ya = F.relu(self.wloc1a(ya))
        ya = nn.MaxPool2d(2)(ya)
        ya = F.relu(self.wloc2a(ya))
        ya = nn.MaxPool2d(2)(ya)


        yb = F.relu(self.wloc1b(w_x_polo['out']))
        yb = nn.MaxPool2d(2)(yb)        
        yb = F.relu(self.wloc2b(yb))
        yb = nn.MaxPool2d(2)(yb)
        
        yc = F.relu(self.wloc2c(w_x_polo['in']))
        yc = nn.MaxPool2d(2)(yc)

        y = torch.cat((ya, yb, yc), dim=2)
        y = F.relu(self.wloc3(y))
        y = nn.MaxPool2d(2)(y)
        y = F.relu(self.wloc4(y))
        y = nn.MaxPool2d((3,2))(y)

        y = F.relu(self.wloc5(y.view(-1, 1000 * (((n_levels['in']-1) * n_eccentricity['in'] * 3) // 8 * n_azimuth['in'] // 8))))
        y = self.wloc6(y)

        return y, theta, z

def train(epoch, loader, n_sample_train):
    model.train()
    train_loss = 0
    kl_loss = 0
    entropy = 0
    correct = 0
    if n_sample_train is None:
        n_sample_train = len(image_dataset['train']) // args.batch_size
        print(f'n_sample_train : {n_sample_train}')
    for num_batch, (data, target) in enumerate(loader):

        data_original, data_polo = data[0], data[1]
        data_original = data_original.to(device, dtype=torch.double)
        data_polo['in'] = data_polo['in'].to(device, dtype=torch.double)
        data_polo['out'] = data_polo['out'].to(device, dtype=torch.double)
        data_polo['ext'] = data_polo['ext'].to(device, dtype=torch.double)
        target = target.to(device)
        
        optimizer.zero_grad()
        output, theta, z = model(data_original, data_polo)
        if model.do_stn and model.deterministic:
            loss = loss_func(output, target) + kl_divergence(model, z) #+ negentropy_loss(model, z)
        else:
            loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        if True: #batch_idx % args.log_interval == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKL Loss: {:.6f}\tEntropy bonus: {:.6f}'.format(
                epoch, 
                args.epochs, 
                (num_batch+1) * args.batch_size,
                n_sample_train * args.batch_size,
                100. * (num_batch+1) / n_sample_train, 
                loss_func(output, target).item(), 
                kl_divergence(model, z).item(),
                -negentropy_loss(model, z).item()))
            print(f'Correct :{100 * pred.eq(target.view_as(pred)).sum().item() / args.batch_size}')
        train_loss += loss_func(output, target).item()
        kl_loss += kl_divergence(model, z).item()
        entropy -= negentropy_loss(model, z).item()
        # get the index of the max log-probability
        #pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        if num_batch == n_sample_train - 1:
            break
    train_loss /= n_sample_train
    kl_loss /= n_sample_train
    entropy /= n_sample_train
    correct /= n_sample_train * args.batch_size
    return correct, train_loss, kl_loss, entropy

def test(loader, n_sample_test):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        kl_loss = 0
        entropy = 0
        correct = 0 
        if n_sample_test is None:
            n_sample_test = len(image_dataset['test']) // args.batch_size
        model.deterministic = True
        for num_batch, (data, target) in enumerate(loader):
            data_original, data_polo = data[0], data[1]
            data_original = data_original.to(device, dtype=torch.double)            
            data_polo['in'] = data_polo['in'].to(device, dtype=torch.double) 
            data_polo['out'] = data_polo['out'].to(device, dtype=torch.double) 
            data_polo['ext'] = data_polo['ext'].to(device, dtype=torch.double)
            target = target.to(device)

            output, theta, z = model(data_original, data_polo)

            # sum up batch loss
            #test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += loss_func(output, target).item()
            kl_loss += kl_divergence(model, z).item()
            entropy -= negentropy_loss(model, z).item()
            # get the index of the max log-probability
            #pred = output.max(1, keepdim=True)[1]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            if num_batch == n_sample_test - 1:
                break

        test_loss /= n_sample_test
        kl_loss /= n_sample_test
        entropy /= n_sample_test
        correct /= n_sample_test * args.batch_size
        print('\nTest set: CE loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), KL loss: {:.4f}, Entropy: {:.4f}\n'.
              format(test_loss, 
                     correct * n_sample_test * args.batch_size, 
                     n_sample_test * args.batch_size,
                     100. * correct,
                     kl_loss, 
                     entropy)
                )
        return correct, test_loss, kl_loss, entropy


lr = 1e-4
LAMBDA = 0
deterministic = True
do_stn = True
radius = 0.5

if __name__ == '__main__':


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = torch.load("../models/low_comp_polo_stn.pt")
    what_model = Polo_AttentionTransNet(LAMBDA=LAMBDA, deterministic=deterministic).to(device)
    what_model = torch.load(f"out/230302bis_polo_stn_dual_WHAT_0.5_True.pt",
                            map_location=torch.device('cpu'))
    
    selected_params = {'wloc0.weight', 'wloc0.bias',
                       'wloc1a.weight', 'wloc1a.bias',
                       'wloc1b.weight', 'wloc1b.bias',
                       'wloc2a.weight', 'wloc2a.bias',
                       'wloc2b.weight', 'wloc2b.bias',
                       'wloc2c.weight', 'wloc2c.bias',
                       'wloc3.weight', 'wloc3.bias',
                       'wloc4.weight', 'wloc4.bias',
                       'wloc5.weight', 'wloc5.bias',
                       'wloc6.weight', 'wloc6.bias'}      
         
    what_params = {k: v for k, v in what_model.state_dict().items() if k in selected_params}
    
    model = Polo_ChanAttentionTransNet(LAMBDA=LAMBDA).to(device)
    model.load_state_dict(what_params, strict=False)    

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9, last_epoch=-1) #, verbose=True)

    # In[35]:

    args.epochs = 300
    args.radius = radius

    train_acc = []
    train_loss = []
    train_kl_loss = []
    train_entropy = []
    test_acc = []
    test_loss = []
    test_kl_loss = []
    test_entropy = []

    for epoch in range(args.epochs):
        n_sample_train = 1
        n_sample_test = 1
        
        params = []
        if epoch % 2 == 1:
            lr = 3e-6
            model.deterministic=True            
            params.extend(list(model.loc_ext_1.parameters()))
            params.extend(list(model.loc_ext_2.parameters()))
            params.extend(list(model.loc_ext_3.parameters()))
            params.extend(list(model.loc_out_1.parameters()))
            params.extend(list(model.loc_out_2.parameters()))
            params.extend(list(model.loc_in_1.parameters()))
            params.extend(list(model.loc_in_2.parameters()))
            params.extend(list(model.loc_in_out.parameters()))
            params.extend(list(model.loc_all.parameters()))
            params.extend(list(model.fc_where.parameters()))
            params.extend(list(model.mu.parameters()))
            optimizer = optim.Adam(params, lr=lr)
        else:
            lr = 1e-4
            model.deterministic=False
            params.extend(list(model.wloc0.parameters()))
            params.extend(list(model.wloc1a.parameters()))
            params.extend(list(model.wloc1b.parameters()))
            params.extend(list(model.wloc2a.parameters()))
            params.extend(list(model.wloc2b.parameters()))
            params.extend(list(model.wloc2c.parameters()))
            params.extend(list(model.wloc3.parameters()))
            params.extend(list(model.wloc4.parameters()))
            params.extend(list(model.wloc5.parameters()))
            params.extend(list(model.wloc6.parameters()))
            optimizer = optim.Adam(params, lr=lr)
        
        acc, loss, kl_loss, entropy = train(epoch, dataloader['train'], n_sample_train)
        train_acc.append(acc)
        train_loss.append(loss)
        train_kl_loss.append(kl_loss)
        train_entropy.append(entropy)

        acc, loss, kl_loss, entropy = test(dataloader['test'], n_sample_test)
        test_acc.append(acc)
        test_loss.append(loss)
        test_kl_loss.append(kl_loss)
        test_entropy.append(entropy)
        torch.save(model, f"out/230320b_polo_stn_dual_{LAMBDA}_sample.pt")
        with open(f"out/230320b_polo_stn_dual_{LAMBDA}_sample.pkl", "wb") as f:
            train_data = {
                "train_acc" : train_acc,
                "train_loss" : train_loss,
                "train_kl_loss" : train_kl_loss,
                "train_entropy" : train_entropy,
                "test_acc" : test_acc,
                "test_loss" : test_loss,
                "test_kl_loss" : test_kl_loss,
                "test_entropy" : test_entropy
                }
            pickle.dump(train_data, f)
        
    model.cpu()
    torch.cuda.empty_cache()

