import numpy as np
import os
import cv2
import random
import pickle
y = []	# labels
X = []	# images 
P = []
i = 0
folderPredix = '/Users/JudyLu/Desktop/2017fall_smile/src/images/'
cls  =['Anger','Disgust','Fear','Happy','Neutral','Sadness','Surprise']
for cl  in cls:
    foldLoc = folderPredix+cl+'/'
    for im in  os.listdir(foldLoc):  

        f = foldLoc+im
        
   
        if im[0] == '.':
            continue  
        
        img = cv2.imread(f,0) #greyscale
        print(img.shape)
        # if img.shape != (256,256)
        # 	print (im)
        img = cv2.resize(img, (256, 256))
        P.append(img)

        print (img.shape)

        #resize by multiplying 0.5
        # res = cv2.resize(img,None,fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
        #new image size: 64x64
        imM = np.array(img)
        X.append(imM)
        y.append(i)
        P.append(im)
       

    i=i+1

for a in range(10):
	print (P[a])
	
	cv2.imshow('',P[a])
cv2.waitKey()  

                                       



n = len(y)
idx = [i for i in range(n)]
# for id in idx:
# 	if X[id].size() != [256,256]
# 	print y[id]

	

	