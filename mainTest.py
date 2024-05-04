import cv2   #นำเข้า OpenCV library เพื่อใช้ในการทำงานกับภาพ
from keras.models import load_model # จากไลบรารี Keras นำเข้าฟังก์ชัน load_model เพื่อโหลดโมเดลที่ถูกtrain
from PIL import Image #: นำเข้าคลาส Image จากไลบรารี PIL เพื่อใช้ในการโหลดและปรับขนาดภาพ
import numpy as np  #นำเข้าโมดูล NumPy สำหรับการประมวลผลข้อมูลแบบตารางหรือ array

model=load_model('BrainTumor10EpochsCategorical.h5')  #โหลดโมเดลที่ถูก train เพื่อตรวจจับเซลล์เนื้องอกในสมอง
image=cv2.imread('D:\Brain tumor dataset\pred\pred5.jpg') #อ่านภาพที่ต้องการ predict โดยใช้ฟังก์ชัน imread และนำมาเก็บไว้ในตัวแปร image

img=Image.fromarray(image)
img=img.resize((64,64))  #ปรับขนาดภาพให้มีขนาด 64x64 pixel โดยใช้เมธอด resize
img=np.array(img)  # แปลงภาพกลับเป็น NumPy array เพื่อนำไปใช้ในการ predict

input_img= np.expand_dims(img,axis=0)
result = model.predict(input_img) #predict ผลลัพธ์ โดยใช้โมเดลที่โหลดมาแล้ว
predicted_class = np.argmax(result)  #ใช้ฟังก์ชัน argmax จาก NumPy เพื่อหาดัชนีของค่าสูงสุด
print(predicted_class) # พิมพ์คลาสที่ถูกทำนายออกทางหน้าจอ


