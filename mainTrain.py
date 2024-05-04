import cv2
import os  # เรียกใช้โมดูล os เพื่อจัดการไฟล์
import tensorflow as tf # เรียกใช้ TensorFlow
from tensorflow import keras # เรียกใช้โมดูล Keras จาก TensorFlow
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split  # เรียกใช้ฟังก์ชัน train_test_split เพื่อแบ่งข้อมูลเป็นข้อมูลที่นำมา train และข้อมูลที่นำมา test
# เรียกใช้ฟังก์ชันจาก Keras
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical



image_directory = 'datasets/' # กำหนดตัวแปรที่ใช้ในการจัดเก็บชื่อไฟล์รูปภาพและไดเรกทอรี
no_tumor_images = os.listdir(image_directory+'no/')
yes_tumor_images = os.listdir(image_directory+'yes/')
dataset=[] # เก็บข้อมูลรูปภาพ
label=[] # เก็บป้ายชื่อว่าภาพเป็น no tumor หรือ yes tumor

INPUT_SIZE=64  # กำหนดขนาดของรูปภาพที่เข้ามาในโมเดล

#print(no_tumor_images)
#path = 'no0.jpg'
#print(path.split('.')[1])

# อ่านและปรับขนาดภาพให้อยู่ในขนาด INPUT_SIZE x INPUT_SIZE

for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'): # เช็คนามสกุลของไฟล์ภาพ
        image=cv2.imread(image_directory+'no/'+image_name) # อ่านภาพจากไฟล์
        image=Image.fromarray(image,'RGB') # แปลงเป็นรูปแบบของภาพจาก NumPy array เป็น PIL image
        image=image.resize((INPUT_SIZE,INPUT_SIZE)) # ปรับขนาดภาพ
        dataset.append(np.array(image))  # เพิ่มข้อมูลภาพลงใน dataset
        label.append(0)  # เพิ่มป้ายชื่อ no tumor


for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)  # เพิ่มป้ายชื่อ yes tumor


dataset=np.array(dataset) # แปลง dataset เป็น NumPy array
label=np.array(label)  # แปลง label เป็น NumPy array

x_train,x_test,y_train,y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)
x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

y_train=to_categorical(y_train , num_classes = 2)
y_test=to_categorical(y_test , num_classes = 2)

#Model Building
#64,64,3
model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(INPUT_SIZE,INPUT_SIZE,3)))  # เพิ่มชั้น Convolutional แบบ 2 มิติ
model.add(Activation('relu'))  # เพิ่มฟังก์ชัน Activation
model.add(MaxPooling2D(pool_size=(2,2))) # เพิ่มชั้น MaxPooling เพื่อลดขนาดของภาพ

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform')) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # แปลงข้อมูลเข้ามาเป็นเวกเตอร์แบบ 1 มิติ
model.add(Dense(64)) # เพิ่มชั้น Dense โดยมี 64 units
model.add(Activation('relu'))
model.add(Dropout(0.5)) # เพิ่มชั้น Dropout เพื่อป้องกันการ overfitting
model.add(Dense(2))  # เพิ่มชั้น Dense โดยมี 2 units
model.add(Activation('softmax'))  # เพิ่มฟังก์ชัน Activation Softmax

#Binary CrossEntropy=1, sigmoid
#Cross Entropy=2 , softmax

model.compile(loss='categorical_crossentropy' , optimizer= 'adam' , metrics = ['accuracy'])

model.fit(x_train,y_train,  #train โมเดลด้วย ข้อมูลที่นำมา train และข้อมูลที่นำมา test
          batch_size=16,
          verbose=1,epochs=10,
          validation_data=(x_test,y_test),
          shuffle=False)

model.save('BrainTumor10EpochsCategorical.h5') # บันทึกโมเดลเมื่อเทรนเสร็จสิ้น







