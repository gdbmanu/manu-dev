#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing libraries
import argparse
import json
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
import numpy as np
from nltk.corpus import wordnet as wn
#from numpy import random
import os
import requests
import time

from time import strftime, gmtime
datetag = strftime("%Y-%m-%d", gmtime())

HOST, device = os.uname()[1], torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# to store results
import pandas as pd


# In[7]:


import seaborn as sns
import sklearn.metrics
from scipy import stats
from scipy.special import logit, expit


# In[4]:


import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from skimage.transform import warp_polar


# In[5]:


colors = ['b', 'r', 'k', 'g', 'm', 'y']
fig_width = 20
phi = (np.sqrt(5)+1)/2 # golden ratio for the figures :-)


# In[8]:


# normalization used to train VGG
# see https://pytorch.org/hub/pytorch_vision_vgg/
v_mean = np.array([0.485, 0.456, 0.406])
v_std = np.array([0.229, 0.224, 0.225])
transforms_norm = transforms.Normalize(mean=v_mean, std=v_std) # to normalize colors on the imagenet dataset


# In[12]:


image_size = 256

data_transform =  transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.AutoAugment(), # https://pytorch.org/vision/master/transforms.html#torchvision.transforms.AutoAugment
            transforms.ToTensor(),      # Convert the image to pyTorch Tensor data type.
            transforms_norm, 
            #to_log_polar(int(image_size))
        ])


# In[36]:


image_path = "data/animal/"

image_dataset = { 'train' : datasets.ImageFolder(
                            image_path+'train', 
                            transform=data_transform
                        ),
                  'test' : datasets.ImageFolder(
                            image_path+'test', 
                            transform=data_transform
                        )
                }


# In[48]:


dataset_size = {'train' : len(image_dataset['train']),
                'test' : len(image_dataset['test'])}

dataset_size['train'], dataset_size['test']


# In[40]:


batch_size = 32
num_workers = 1

dataloader = { 'train' : torch.utils.data.DataLoader(
                            image_dataset['train'], batch_size=batch_size,
                            shuffle=True, 
                            num_workers=num_workers
                        ),
               'test' : torch.utils.data.DataLoader(
                            image_dataset['test'], batch_size=batch_size,
                            shuffle=True, 
                            num_workers=num_workers
                        )
             }


# In[41]:


from PIL import ImageFile
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[42]:


criterion = nn.BCEWithLogitsLoss() #binary_cross_entropy_with_logits
n_output = 1
model_path = 'models/re-trained_'
model_name = 'animal'
model_filename = model_path + model_name + '.pt'
results_filename = f'results/{datetag}_{HOST}_train_{model_name}.json'


# In[43]:


results_filename


# In[44]:


model = torchvision.models.vgg16(pretrained=True) 


# In[45]:


num_features = model.classifier[-1].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, n_output)]) # Add our layer
model.classifier = nn.Sequential(*features) # Replace the model classifier


# In[30]:


model.to(device)


# In[33]:


lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, amsgrad=amsgrad)


# In[34]:


df_train = pd.DataFrame([], columns=['epoch', 'avg_loss', 'avg_acc', 'avg_loss_val', 'avg_acc_val', 'device_type']) 


# In[46]:


dataloader['train']


# In[ ]:


num_epochs = 1

for epoch in range(num_epochs):
    loss_train = 0
    acc_train = 0
    for i, (images, labels) in enumerate(dataloader['train']):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs[:,0], labels.float())

        loss.backward()
        optimizer.step()

        loss_train += loss.item() * images.size(0)
        preds = torch.round(torch.sigmoid(outputs[:,0].data))
        acc_train += torch.sum(preds == labels.data)

    avg_loss = loss_train / dataset_size['train']
    avg_acc = acc_train / dataset_size['train']

    with torch.no_grad():
        loss_val = 0
        acc_val = 0
        for i, (images, labels) in enumerate(dataloader['test']):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs[:,0], labels.float())

            loss_val += loss.item() * images.size(0)
            preds = torch.round(torch.sigmoid(outputs[:,0].data))
            acc_val += torch.sum(preds == labels.data)

        avg_loss_val = loss_val / dataset_size['test']
        avg_acc_val = acc_val / dataset_size['test']

    df_train.loc[epoch] = {'epoch':epoch, 'avg_loss':avg_loss, 'avg_acc':float(avg_acc),
                           'avg_loss_val':avg_loss_val, 'avg_acc_val':float(avg_acc_val), 'device_type':device.type}
    print(f"Epoch {epoch+1}/{num_epochs} : train= loss: {avg_loss:.4f} / acc : {avg_acc:.4f} - val= loss : {avg_loss_val:.4f} / acc : {avg_acc_val:.4f}")

model.cpu()
torch.cuda.empty_cache()


# In[ ]:




