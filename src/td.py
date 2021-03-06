import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import utils
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import cv2
import os
import random
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models import *
from utils import progress_bar
from torch.autograd import Variable
use_gpu = False

# folderPredix = '/Users/JudyLu/Desktop/2017fall_smile/src/images/'
# cls  =['Anger','Disgust','Fear','Happy','Neutral','Sadness','Surprise']
#        # 0     # 1        # 2  # 3        #4       # 5         # 6

# ima = []
# name = []
# i = 0
# y = []
# X = []
# for cl  in cls:
#     foldLoc = folderPredix+cl+'/'
#     for im in  os.listdir(foldLoc): #picture name, eg: Anger185.jpg.jpg
#         f = foldLoc+im #path of each picture, eg: /Users/JudyLu/Desktop/2017fall_smile/src/images/Anger/Anger185.jpg.jpg
#         if im[0] == '.':
#             continue  
#         img = cv2.imread(f,0) #greyscale,matrix form of an image
#         ima.append(img)
#         name.append(im)
#         res = cv2.resize(img,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
#         imM = np.array(res) 
#         X.append(imM)
#         y.append(i)
# i=i+1


# with open('data.pickle', 'rb') as f:
#     data  = pickle.load(f)

# X = np.array( data['XTr']/255.0, dtype ='f') #normalize#images
# y = data['yTr'] #labels  
# nameindex = data['N']                      #training images
# yTe = data['yTe']
# xTe =np.array( data['XTe']/255.0, dtype='f')
# print (len(X), len(xTe))#Train: 2046 Test: 341 

## ==== data part === 


with open('data.pickle', 'rb') as f:
    data  = pickle.load(f)
    train_X  = np.array( data['train_X']/255.0, dtype ='f')
    train_y  = data['train_y ']
    test_X  = np.array( data['test_X']/255.0, dtype ='f')
    test_y  = data['test_y']

train = data_utils.TensorDataset(train_X, train_y)
train_loader = data_utils.DataLoader(train, batch_size=20, shuffle=True)

test = data_utils.TensorDataset(test_X, test_y)
test_loader = data_utils.DataLoader(test, batch_size=20, shuffle=False)
best_acc = 0
use_cuda = torch.cuda.is_available()
start_epoch = 0
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
        

model = Model()

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for idx, (X, y) in enumerate(test_loader):
        if use_cuda:
            X, y = X.cuda(), y.cuda()
        X, y = Variable(X, volatile=True), Variable(y)
        outputs = model(X)
        loss = criterion(outputs, y)

        test_loss += loss.data[0]
        _, predicted = torch.max(y.data, 1)
        total += y.size(0)
        correct += predicted.eq(y.data).cpu().sum()

        progress_bar(idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'Model': Model.module if use_cuda else model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for idx, (X, y) in enumerate(train_loader):
        if use_cuda:
            X, y = X.cuda(), y.cuda()
        optimizer.zero_grad()
        X, y = Variable(X), Variable(y)
        outputs = model(X)
        loss = criterion(X, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(y.data).cpu().sum()

        progress_bar(idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(idx+1), 100.*correct/total, correct, total))
#train the model
# for epoch in range(200):
#     train_accu = []
#     test_accu = []
#     for i in range(n//batch_size): #102
#         print(i)
#         optimizer.zero_grad()
#         yD = y[i*batch_size : (i+1)*batch_size]      
#         print (yD)
#         if use_gpu: 
#             yD = yD.cuda()
#         # print (yD)
#         yy = torch.from_numpy(yD)
#         # print ('label', Variable(yy))
#         train_label = Variable(yy)
#         #print (train_label) eg:[3 5 3 1 4 5 3 6 4 5 0 4 3 5 4 4 1 0 6 2]
#         xD  =  X[i*batch_size:(i+1)*batch_size,:,:,:] 
#         print(nameindex[i*batch_size:(i+1)*batch_size])
       
#         if use_gpu: 
#             xx = xx.cuda()
#         xx = torch.from_numpy(xD) # creates a tensor 
#         xOut = model.forward(Variable(xx))
#         #break
#         loss = F.nll_loss(xOut, train_label)
#         loss.backward()    # calc gradients
#         train_loss.append(loss.data[0])
#         optimizer.step()   # update gradients
#         #print (xOut)
#         Train_prediction = xOut.data.max(1)[1]
#         print(Train_prediction[0])

#         # print(Train_prediction)
#         print ("pre", xOut.data.max(1)[1])
        
#         for a in nameindex[i*batch_size:(i+1)*batch_size]: 
#             print(a)
#             print(name[a])
            
#             cv2.imshow(name[a], ima[a.all()])
#             cv2.waitKey()
#             cv2.destroyAllWindows()
        
#         # for a in nameindex[i*batch_size:(i+1)*batch_size] and b in (0,20): 
#         #     print (a)
#         #     print(ima[a])
#         #     cv2.imshow(Train_prediction[b], ima[a.all()])
#         #     cv2.waitKey()
#         #     cv2.destroyAllWindows()

#         trainAccuracy = Train_prediction.eq(train_label.data).sum()/batch_size*100
#         #print (trainAccuracy)
#         train_accu.append(trainAccuracy)
#         if j % 10 == 0:
#             print('Train Step: {}\t\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(j, loss.data[0], trainAccuracy))
#         j += 1
#     avg_train_accu.append(np.mean(train_accu))
#     print ('epoch')
#     testing_acc()

#print (train_accu)
#print (test_accu)
#x = np.arange(0, 100)
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

plt.plot(avg_train_accu, label = 'training')
plt.plot(avg_test_accu, label = 'testing')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))