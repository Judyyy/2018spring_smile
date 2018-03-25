# @Author: Xi He <Heerye>
# @Date:   2018-03-25T18:01:04-04:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: xi_guide.py
# @Last modified by:   Heerye
# @Last modified time: 2018-03-25T18:03:36-04:00



# coding: utf-8

# In[4]:


# import sys
# assert sys.executable == '/root/miniconda3/bin/python3', print('invalid kernel!')
#
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[5]:


import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import utils

import matplotlib.pyplot as plt
import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import pickle
import cv2
import os
import random

use_gpu = False


# In[6]:


from dataprep import train_X, train_y, test_X, test_y, cls
print(len(train_X),len(test_X))


# In[7]:


transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class Emotional(torch.utils.data.TensorDataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = (self.X[idx]/255.0).astype(float)

#         if self.transform:
#             image = self.transform(transform)

        sample = [image, self.y[idx]]

        return sample


# In[8]:


train = Emotional(train_X, train_y, transform=transform)
test = Emotional(test_X, test_y, transform=transform)


# In[9]:


train_loader = torch.utils.data.DataLoader(train, batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=20, shuffle=False)


# In[10]:


for idx, (x, y) in enumerate(train_loader):
    print(x.shape, type(x), y.shape, type(y))
    for i in range(len(x)):
        plt.subplot(4, len(x)/4, i+1)
        plt.subplots_adjust(wspace=1, hspace=1)
        plt.imshow(x[i].numpy()[0])
        plt.axis('off')
        plt.title(cls[y[i]])
    if idx == 0:
        break

plt.show()


# In[46]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # input is 128x128
        # padding=2
        self.conv1 = nn.Conv2d(1, 96, 7, padding = 2)
        self.conv2 = nn.Conv2d(96, 256, 5, padding = 2)
        self.conv3 = nn.Conv2d(256, 512, 3, padding = 2)
        self.conv4 = nn.Conv2d(512, 512, 3, padding = 2)
        self.conv5 = nn.Conv2d(512, 512, 3, padding = 2)

        self.fc1 = nn.Linear(512*16*16, 4048)
        self.fc2 = nn.Linear(4048, 4049) #1024
        self.fc3 = nn.Linear(4049, 7)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), 3)
        # return x
        x = x.view(-1, 512*16*16)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        print(x)
        return F.log_softmax(x)

class NaiveModel(nn.Module):
    def __init__(self):
        super(NaiveModel, self).__init__()
        # input is 64x64
        # padding=2
        self.conv1 = nn.Conv2d(1, 4, 9, stride = 2, padding = 2)

        self.fc1 = nn.Linear(4*10*10, 7)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3)

        x = x.view(-1, 4*10*10)   # reshape Variable
        x = F.relu(self.fc1(x))
        return F.log_softmax(x)


# In[47]:


model = NaiveModel()


# In[48]:


if use_gpu:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# In[49]:


print(model)


# In[58]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# In[59]:


def train(epoch):
    print('Epoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.float()
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# In[62]:


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.float()
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


# In[64]:


start_epoch = 0
for epoch in range(start_epoch, start_epoch+10):
    train(epoch)
    test(epoch)
