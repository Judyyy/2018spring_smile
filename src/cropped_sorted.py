import glob
from shutil import copyfile # copy file from one to another one
from shutil import move
import cv2
import sys
import os
from random import *
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
participants = glob.glob("./images/*") 
#print(participants)

for x in participants:
    #print("%s"%x)

    for sessions in glob.glob("%s/*"%x):

        
        #print(part)        
        
        image = cv2.imread(sessions)
        

        # resized_image = cv2.resize(image, (256, 256))
        # # # cvt_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(sessions+".jpg",resized_image)
        # os.remove(sessions)

        #print (sessions)
        print(image.shape)
        
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        
        # #for i, face in enumerate(image_faces):
        # #   cv2.imwrite("face-" + str(i) + ".jpg", face)

        # for (x,y,w,h) in faces:
           

        #     # w=(x+w-257)
        #     # h=(y+h-257)

        #     # if w>x and h>y :
            
        #     #     a = randint(x,w)
        #     #     b = randint(y,h)

        #     # else:
        #     #     a
        #     #     b
            
           
        #     image = image[y: y+h , x: x+w]
        #     print(image.shape)
            # cv2.imwrite(sessions+"new.jpg",image)
            # os.remove(sessions)
            #os.rename(sessions+".jpg",sessions)    
            #copyfile(image, "./sorted_set/%s/%s"%(emotions[emotion], part))    


           
          
		

