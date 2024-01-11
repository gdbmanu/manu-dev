#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, models, transforms
from typing import List, Tuple

from easydict import EasyDict as edict
import pickle

# In[21]:


args = edict({})
args.image_size = 240
args.batch_size = 40
args.log_interval = 100
args.std_sched = .3


# In[6]:


# normalization used to train VGG
# see https://pytorch.org/hub/pytorch_vision_vgg/
v_mean = np.array([0.485, 0.456, 0.406])
v_std = np.array([0.229, 0.224, 0.225])
transforms_norm = transforms.Normalize(mean=v_mean, std=v_std) # to normalize colors on the imagenet dataset


# In[7]:


transform_base =  transforms.Compose([
            transforms.Resize((int(args.image_size), int(args.image_size))),
            transforms.ToTensor(),  # Convert the image to pyTorch Tensor data type.
            transforms_norm, ])      
    


# In[8]:

#image_path = "/envau/work/brainets/dauce.e/data/Imagenet/"
image_path = "/home/INT/dauce.e/data/Imagenet_full/"
#image_path = "/media/manu/Seagate Expansion Drive/Data/Imagenet/"
#image_path = "/run/user/1001/gvfs/sftp:host=bag-008-de03/envau/work/brainets/dauce.e/data/Imagenet/"
#image_path = "../data/animal/"
#image_path = "../animal/"


image_dataset = { 'train' : datasets.ImageFolder(
                            image_path+'train',
                            transform=transform_base
                        ),
                  'test' : datasets.ImageFolder(
                            image_path+'val',
                          transform=transform_base
                        )
                }


# In[9]:


num_workers = 1

dataloader_orig = { 'train' : torch.utils.data.DataLoader(
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






# Créez un DataLoader pour le sous-ensemble en utilisant SubsetRandomSampler
from torch.utils.data import SubsetRandomSampler
n_labels = 1000
if n_labels < 1000:
    # Créez un sous-ensemble en sélectionnant les indices des échantillons que vous souhaitez inclure
    train_size = 1300 * n_labels
    test_size = 50 * n_labels
    labels = np.random.permutation(1000)[:n_labels]
    train_indices = []
    test_indices = []
    for i in labels:
        train_indices += list(1300 * i + np.arange(1300))
        test_indices += list(50 * i + np.arange(50)) 

    subset_indices = {'train': train_indices, 'test': test_indices}
    print(labels, test_indices)

    dataloader = {}
    for cat in ('train', 'test'):
        dataloader[cat] = torch.utils.data.DataLoader(
            dataset=dataloader_orig[cat].dataset,  # Utilisez le même ensemble de données que le DataLoader d'origine
            batch_size=dataloader_orig[cat].batch_size,
            sampler=SubsetRandomSampler(subset_indices[cat]),  # Utilisez SubsetRandomSampler avec les indices du sous-ensemble
            num_workers=dataloader_orig[cat].num_workers,
            pin_memory=dataloader_orig[cat].pin_memory,
            )
else:
    labels = range(1000)
    train_indices = []
    test_indices = []
    for i in labels:
        train_indices += list(100 * i + np.arange(100))
        test_indices += list(10 * i + np.arange(10)) 

    subset_indices = {'train': train_indices, 'test': test_indices}
    dataloader = dataloader_orig


def expand_dim(tensor, dim, desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))


# In[14]:


def negentropy_loss(model, z):
    z_mean = torch.mean(z, dim=0)
    z_std = torch.std(z, dim=0)
    p = torch.distributions.Normal(torch.ones_like(z)*z_mean, torch.ones_like(z) * z_std)
    return p.log_prob(z).sum()

def kl_divergence(model, z):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    z_dims = z.size()
    if args.radius > 0:
        #p = torch.distributions.Normal(torch.zeros_like(z), args.radius * torch.ones_like(z))
        p = torch.distributions.MultivariateNormal(torch.zeros_like(z), 
                                                   args.radius * torch.einsum('i,jk->ijk',torch.ones(z_dims[0]), torch.eye(2)))
    else:
        p = torch.distributions.Normal(torch.zeros_like(z), 1e-6 * torch.ones_like(z))


    # 2. get the probabilities from the equation
    #log_qzx = model.q.log_prob(z)
    log_pz = p.log_prob(z)

    z_mean = torch.mean(z, dim=0)
    z_std = torch.std(z, dim=0) + 1e-6
    #print(z)
    #print(torch.cov(z.T))
    #print(torch.eye(2))
    z_cov = torch.cov(z.T) + 1e-6 * torch.eye(2)
    #print(z_std)
    #q = torch.distributions.MultivariateNormal(torch.ones_like(z)*z_mean, torch.ones_like(z) * z_std)
    q = torch.distributions.MultivariateNormal(torch.ones_like(z)*z_mean, torch.ones(z_dims[0], 2, 2) * z_cov)
    log_qzx = q.log_prob(z)

    # kl
    #print(log_pz)
    #print(log_qzx)
    kl = (log_qzx - log_pz)
    
    # sum over last dim to go from single dim distribution to multi-dim
    kl = model.LAMBDA * kl.sum()
    return kl

# In[22]:


class Grid_AttentionTransNet(nn.Module):
    
    def __init__(self, do_stn=True, do_what=False, LAMBDA=.1, deterministic=False):
        super(Grid_AttentionTransNet, self).__init__()
        
        self.do_stn = do_stn
        self.do_what = do_what
        self.deterministic = deterministic
        self.LAMBDA = LAMBDA
        
        self.resnet = models.resnet101(pretrained=True) 
        #self.resnet.train():
        self.resnet_where = models.resnet101(pretrained=True) 
        self.resnet_wmult = models.resnet101(pretrained=True) 
        #self.resnet_where.eval():
        
        ##  The what pathway
        
        #self.num_features = self.vgg.classifier[-1].in_features
        #features = list(self.vgg.classifier.children())[:-1] # Remove last layer
        #features.extend([nn.Linear(num_features, 500)]) # Add our layer
        #self.vgg.classifier = nn.Sequential(*features) # Replace the model classifier
        
        self.what_grid = self.logPolarGrid(-1,-6) 
        #self.what_grid = self.logPolarGrid(0,-5) 
        
        
        #self.fc_what = nn.Linear(self.num_features, 1000)
        #self.fc_what.weight.data /= torch.sqrt(n_features)
        #self.fc_what.bias.data /= torch.sqrt(n_features)

        ##  The where pathway        
        self.num_features = self.resnet_where.fc.in_features
        n_features = torch.tensor(self.num_features, dtype=torch.float)
        
        features = list(self.resnet_where.children())[:-1] # Remove last layer
        #features[7][2].relu = nn.Identity()
        #features.extend([nn.Linear(num_features, 500)]) # Add our layer
        self.resnet_where = nn.Sequential(*features) # Replace the model classifier

        features = list(self.resnet_wmult.children())[:-1] # Remove last layer
        features[7][2].relu = nn.Identity()
        #features.extend([nn.Linear(num_features, 500)]) # Add our layer
        self.resnet_wmult = nn.Sequential(*features) # Replace the model classifier
        
        self.where_grid = self.logPolarGrid(0,-5)
        
        self.mu = nn.Linear(self.num_features, 2) #, bias=False)
        self.logvar = nn.Linear(self.num_features, 2) #, bias=False)
        
        self.mu.weight.data /= torch.sqrt(n_features) 
        self.mu.bias.data /= torch.sqrt(n_features)
        
        self.logvar.weight.data /= torch.sqrt(n_features)
        self.logvar.bias.data /= torch.sqrt(n_features)

        self.identity = nn.Parameter(torch.tensor([[1, 0], [0, 1]],
                                                   dtype=torch.float),
                                      requires_grad=False)
        self.downscale = nn.Parameter(torch.tensor([[0.33, 0], [0, 0.33]],
                                                   dtype=torch.float),
                                      requires_grad=False)
        self.dropout = torch.nn.Dropout()
    
    def logPolarGrid(self, a, b, base=2):
        rs = torch.logspace(a, b, args.image_size, base = base)
        ts = torch.linspace(0, torch.pi*2, args.image_size)
        
        grid_xs = torch.outer(rs, torch.cos(ts)) 
        grid_xs = grid_xs.unsqueeze(0).unsqueeze(3)
        grid_xs = Variable(grid_xs, requires_grad=True)

        grid_ys = torch.outer(rs, torch.sin(ts)) 
        grid_ys = grid_ys.unsqueeze(0).unsqueeze(3)
        grid_ys = Variable(grid_ys,  requires_grad=True)
        
        grid = torch.cat((grid_xs, grid_ys), 3)
        grid = expand_dim(grid, 0, args.batch_size)
        grid.requires_grad_(True)
        
        return grid.to(device)    

    def stn(self: object, x: torch.Tensor) -> Tuple[torch.Tensor]:
    
        logPolx = x #F.grid_sample(x, self.where_grid)
        
        if self.do_stn:
            with torch.no_grad():
                y_mult = self.resnet_wmult(logPolx).view(-1, self.num_features)
                y_mult = nn.Hardsigmoid()(10 * (y_mult-1))
                #print(y_mult)
                #print(torch.mean(y_mult),torch.std(y_mult),torch.min(y_mult),torch.max(y_mult))
                #y_mult_mult = torch.prod(y_mult, dim=0)
                #print(torch.mean(y_mult_mult),torch.std(y_mult_mult),torch.min(y_mult_mult),torch.max(y_mult_mult))
            if True: #with torch.no_grad():
                y_where = self.resnet_where(logPolx).view(-1, self.num_features)
                #print(torch.mean(y_where),torch.std(y_where),torch.min(y_where),torch.max(y_where))
            y = y_where * y_mult
            #print(torch.mean(y),torch.std(y),torch.min(y),torch.max(y))
            
            mu = self.mu(y)
                               
            if self.deterministic:
                sigma = args.radius * torch.ones_like(mu)
                self.q = torch.distributions.Normal(mu, sigma)  
                z = mu
            else:
                logvar = self.logvar(y) + 6
                sigma = torch.exp(-logvar / 2)
                self.q = torch.distributions.Normal(mu, sigma)      
                z = self.q.rsample()
            theta = torch.cat((self.identity.unsqueeze(0).repeat(
                                z.size(0), 1, 1), z.unsqueeze(2)),# !!!
                                  dim=2)
            print(theta[0,...])
            print(z[0,...])
        
            grid_size = torch.Size([x.size()[0], x.size()[1], args.image_size, args.image_size])
            self.shift_grid = F.affine_grid(theta, grid_size)
            self.shift_grid.requires_grad_(True)
            x = F.grid_sample(x, self.shift_grid)

        else:
            mu = torch.tensor([0, 0],dtype=torch.float)
            mu = mu.unsqueeze(0).repeat(x.size()[0], 1)   
            sigma = torch.tensor([1, 1],dtype=torch.float)
            sigma = sigma.unsqueeze(0).repeat(x.size()[0], 1)    
            
            if self.do_what:
                self.q = torch.distributions.Normal(mu, args.radius * sigma)
                z = self.q.rsample().to(device)
                #z = torch.FloatTensor(z).to(device)
                print(z[0,...])
                theta = torch.cat((self.downscale.unsqueeze(0).repeat(
                                z.size(0), 1, 1), z.unsqueeze(2)),
                                  dim=2)
        
                grid_size = torch.Size([x.size()[0], x.size()[1], args.image_size, args.image_size])
                grid = F.affine_grid(theta, grid_size)
                x = F.grid_sample(x, grid)
            else:
                z = torch.tensor([0, 0],dtype=torch.float)
                z = z.unsqueeze(0).repeat(x.size()[0], 1)

                theta = nn.Parameter(torch.tensor([[1, 0, 0], [0, 1, 0]],
                                                    dtype=torch.float),
                                        requires_grad=False)
                theta = theta.unsqueeze(0).repeat(x.size()[0], 1, 1)
            
           
        return x, theta, z

    def forward(self, x):
        # transform the input
        x_shift, theta, z = self.stn(x)
        
        logPolx = F.grid_sample(x_shift, self.what_grid) # x_shift
        y = self.resnet(logPolx) * label_filter  
        #y = self.fc_what(y)
       
        return y, theta, z



# In[28]:

def train(epoch, loader):
    model.train()
    train_loss = 0
    kl_loss = 0
    entropy = 0
    correct = 0
    try:
      for batch_idx, (data, target) in enumerate(loader):

        data, target = data.to(device, dtype=torch.float), target.to(device)
   
        optimizer.zero_grad()
        output, theta, z  = model(data)
        if model.do_stn and not model.do_what:
            loss = loss_func(output, target) + LAMBDA * kl_divergence(model, z) 
        else:
            loss = loss_func(output, target) #loss_func_contrast(output, output_ref)
        loss.backward()
        #print(model.resnet.fc.weight[:,labels[0]].grad)
        #print(model.resnet.layer1[0].conv1.weight.grad)
        print(model.mu.weight.grad)
        #print(model.shift_grid.grad)
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        if True: #batch_idx % args.log_interval == 0:
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tKL Loss: {:.6f}\tEntropy : {:.6f}'.format(
                epoch, args.epochs, batch_idx * args.batch_size,
                len(subset_indices['train']), #len(dataloader['train'].dataset),
                100. * batch_idx * args.batch_size/len(subset_indices['train']), #/ len(dataloader['train']), 
                loss_func(output, target).item(), 
                kl_divergence(model, z).item(),
                -negentropy_loss(model, z).item()
                ))
            print(f'Correct :{100 * pred.eq(target.view_as(pred)).sum().item() / args.batch_size}')
        train_loss += loss_func(output, target).item()
        kl_loss += kl_divergence(model, z).item()
        entropy -= negentropy_loss(model, z).item()
        # get the index of the max log-probability
        #pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    except:
      pass
    train_loss /= (batch_idx+1)
    kl_loss /= (batch_idx+1)
    entropy /= (batch_idx+1)
    #correct /= len(subset_indices['train']) #len(dataloader['train'].dataset)
    correct /= len(dataloader['train'].dataset)
    return correct, train_loss, kl_loss, entropy


def test(loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        kl_loss = 0
        entropy = 0
        correct = 0
        # model.deterministic = True
        try:
          for n, (data, target) in enumerate(loader):
            data, target = data.to(device, dtype=torch.float), target.to(device)

            output, theta, z = model(data)

            # sum up batch loss
            #test_loss += F.nll_loss(output, target, size_average=False).item()
            test_loss += loss_func(output, target).item()
            kl_loss += kl_divergence(model, z).item()
            entropy -= negentropy_loss(model, z).item()
            # get the index of the max log-probability
            #pred = output.max(1, keepdim=True)[1]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        except:
          pass
        #model.deterministic = deterministic

        test_loss /= (n+1)
        kl_loss /= (n+1)
        entropy /= (n+1)
        test_len = len(dataloader['test'].dataset) #len(subset_indices['test'])
        print('\nTest set: CE loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), KL loss: {:.4f}, Entropy: {:.4f}\n'.
              format(test_loss, correct, test_len,
                     100. * correct / test_len,
                     kl_loss, entropy))
        return correct / test_len, test_loss, kl_loss, entropy

lr =  3e-7 * args.batch_size / 40 #1e-5 #3e-9  
LAMBDA = 0.001 #1e-2 #3e-2 
opt = "Adam"
do_stn = True
do_what = False
deterministic = True

args.epochs = 3000
radius = .1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loss_func =  nn.CrossEntropyLoss()
# loss_func_contrast = nn.MSELoss()
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9, last_epoch=-1) #, verbose=True)
# std_axe = np.linspace(radius * 1/args.epochs, radius, args.epochs)
    
label_filter = torch.zeros(1000).to(device)
for i in labels:
    label_filter[i] = 1


model = Grid_AttentionTransNet(do_stn=do_stn, do_what = do_what, LAMBDA=LAMBDA, deterministic=deterministic).to(device)

save_path = "out/"
f_what = "resnet_focus"
#f_wmult = "resnet_polar_1000"
#f_where = f"231015_ImgNet_logPolarGrid_resnet_stn_{radius}_{LAMBDA}_{deterministic}_jnj_focus_{opt}_{n_labels}_{lr}"
f_name = f"240111_ImgNet_logPolarGrid_resnet_stn_{radius}_{LAMBDA}_{deterministic}_jnj_focus_{opt}_{n_labels}_{lr}"

what_params = torch.load(save_path+f_what+'.pt', map_location=torch.device('cpu'))
#wmult_params = torch.load(save_path+f_wmult+'.pt', map_location=torch.device('cpu'))
    
#selected_params = {'vgg.classifier.0.weight', 'vgg.classifier.0.bias',
#                  'vgg.classifier.3.weight', 'vgg.classifier.3.bias',
#                  'vgg.classifier.6.weight', 'vgg.classifier.6.bias'} 
         
#model_params = {k: v for k, v in saved_params.state_dict().items() if k in selected_params}
#model.load_state_dict(where_params, strict=False)    
model.resnet.load_state_dict(what_params, strict=False)    
#model.resnet_wmult.load_state_dict(wmult_params, strict=False)    
#print(model.resnet)
#exit()
model.LAMBDA = LAMBDA
args.radius = radius
    
params = []
params.extend(list(model.resnet_where.parameters()))
#params.extend(list(model.resnet.parameters()))
params.extend(list(model.mu.parameters()))
if not model.deterministic:
    params.extend(list(model.logvar.parameters()))
if opt == 'SGD':
    optimizer = optim.SGD(params, lr=lr, momentum = 0.9)
else:
    optimizer = optim.Adam(params, lr=lr)

train_acc = []
train_loss = []
train_kl_loss = []
train_entropy = []
test_acc = []
test_loss = []
test_kl_loss = []
test_entropy = []

for epoch in range(args.epochs):
        
    print(f'****** EPOCH : {epoch}/{args.epochs}, radius = {args.radius} ******')
    acc, loss, kl_loss, entropy = train(epoch, dataloader['train'])
    train_acc.append(acc)
    train_loss.append(loss)
    train_kl_loss.append(kl_loss)
    train_entropy.append(entropy)
    acc, loss, kl_loss, entropy = test(dataloader['test'])
    test_acc.append(acc)
    test_loss.append(loss)
    test_kl_loss.append(kl_loss)
    test_entropy.append(entropy)
    where_params = {k for k in model.resnet_where.state_dict()}
    params_to_save = {k: v for k, v in model.resnet_where.state_dict().items() if k in where_params}
    selected_params = {'mu.weight', 'mu.bias', 'logvar.weight', 'logvar.bias'}  
    params_to_save.update({k: v for k, v in model.state_dict().items() if k in selected_params})
    torch.save(params_to_save, save_path + f_name + ".pt")
    with open(save_path + f_name + ".pkl", "wb") as f:
        train_data = {
                "train_acc" : train_acc,
                "train_loss" : train_loss,
                "train_kl_loss" : train_kl_loss,
                "train_entropy" : train_entropy,
                "test_acc" : test_acc,
                "test_loss" : test_loss,
                "test_kl_loss" : test_kl_loss,
                "test_entropy" : test_entropy,
                "active_labels" : labels
                }
        pickle.dump(train_data, f)
    
model.cpu()
torch.cuda.empty_cache()



