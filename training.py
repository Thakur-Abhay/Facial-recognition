import cv2
import os
import numpy as np
from PIL import Image
import pickle
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")
x=10

#face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml') # D:\03 -Python\python37\source_code\Abhay\Face Recognition\cascade\data
face_cascade = cv2.CascadeClassifier('C:\\Users\\abhay\\Dropbox\\My PC (LAPTOP-CPO2A5NR)\\Desktop\AT\opencv tut and face recognition software\\cascades\\data\\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
#model = LBPHFaceRecognizer::create()
#pdb.set_trace()
current_id = 0
label_ids = {}
y_labels = []
x_train = []
temp_cnt=0
for root, dirs, files in os.walk(image_dir):
        
        for file in files:
                #pdb.set_trace()
                if file.endswith("png") or file.endswith("jpeg"):
                        
                        temp_cnt=temp_cnt+1
                        print(temp_cnt)
                        path = os.path.join(root, file)
			# label=os.path.basename(os.path.dirname(path).replace(" ","-").lower())
			# print(path)
                        label = os.path.basename(root).replace(" ", "-").lower()
                        
                        
			#print(label, path)
			# if label in label_ids:
			# 	pass
			# else:
			# 	label_ids[label]=current_id
			# 	current_id+=1
			
                        if not label in label_ids:
                                label_ids[label] = current_id
                                current_id += 1
                        id_ = label_ids[label]
                        
			#print(label_ids)
			
##			y_labels.append(label) # some number
##			x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
                        pil_image = Image.open(path).convert("L") # grayscale
                        size = (550, 550)
                        final_image = pil_image.resize(size, Image.ANTIALIAS)
                        image_array = np.array(final_image, "uint8")#this command converts the image into numbers which we then store in a numpy array
			#print(image_array)#printing the numpy array containing the pixel value of our images
			#
                        faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
			
                        for (x,y,w,h) in faces:
                                roi = image_array[y:y+h, x:x+w]
                                x_train.append(roi)
                                y_labels.append(id_) 
##
##
#print(y_labels)
#print(x_train)
#pdb.set_trace()
with open("C:/Users/abhay/Dropbox/My PC (LAPTOP-CPO2A5NR)/Desktop/AT/opencv tut and face recognition software/face_labels.pickle",'wb') as f:
### f=open("c:/ps/face_labels.pickle", 'wb') 
### f=open("C:/Users/abhay/Dropbox/My PC (LAPTOP-CPO2A5NR)/Desktop/AT/opencv tut and face recognition software/face_labels.pickle",'wb')
        pickle.dump(label_ids, f)
#pdb.set_trace()
recognizer.train(np.array(x_train), np.array(y_labels))
recognizer.save("trainner.yml")
print( ' completed....')
 
