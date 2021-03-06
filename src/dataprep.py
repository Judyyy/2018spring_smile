# @Author: Xi He <Heerye>
# @Date:   2018-03-23T15:30:37-04:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: dataprep.py
# @Last modified by:   Heerye
# @Last modified time: 2018-03-25T17:42:50-04:00



import numpy as np
import os
import cv2
import random
import pickle
import torch
from torch.autograd import Variable

# folderPredix = '/Users/JudyLu/Desktop/2017fall_smile/src/images/'
folderPredix = 'images/'
cls = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', 'Surprise']
# 0     # 1        # 2  # 3        #4       # 5         # 6

y = []  # labels
X = []  # images
i = 0
ima = []
name = []
for cl in cls:
    foldLoc = folderPredix + cl + '/'
    for im in os.listdir(foldLoc):  # picture name, eg: Anger185.jpg.jpg
        f = foldLoc + im  # path of each picture, eg: /Users/JudyLu/Desktop/2017fall_smile/src/images/Anger/Anger185.jpg.jpg
        if im[0] == '.':
            continue
        # print (f)
        img = cv2.imread(f, 0)  # greyscale,matrix form of an image
        ima.append(img)
        name.append(im)
        # resize by multiplying 0.5
        res = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        # new image size: 64x64
        imM = np.array(res)  # matrix form of an image
        # print(imM.shape)
        xx = np.reshape(imM, [1, 64, 64])
        X.append(xx)
        y.append(i)

    i = i + 1

# X = torch.cat([torch.from_numpy(x) for x in X], 0).type(torch.FloatTensor)
# y = torch.from_numpy(np.array(y)).type(torch.LongTensor)

# print(type(X), type(y), X.size(), y.size())
# Todo: permutation index
print(len(X),len(y), 'sss')

def split_data(X, y, ratio=0.8):

    wall = int(len(X) * ratio)
    perm = torch.randperm(len(X))
    train_X = [X[i] for i in perm[:wall]]
    train_y = [y[i] for i in perm[:wall]]
    test_X = [X[i] for i in perm[wall:]]
    test_y = [y[i] for i in perm[wall:]]
    return train_X, train_y, test_X, test_y


train_X, train_y, test_X, test_y = split_data(X, y)

# with open('data.pickle', 'wb') as f:
# 	data={}
# 	data['train_X']=train_X
# 	data['train_y']=train_y
# 	data['test_X']=test_X
# 	data['test_y']=test_y
# 	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
