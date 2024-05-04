import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__) #สร้างอินสแตนซ์ของ Flask โดยใช้ชื่อของโมดูลหรือโปรแกรมเป็นชื่อของแอปพลิเคชัน 


model =load_model('BrainTumor10Epochs.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo): #ฟังก์ชันที่คืนชื่อของคลาสตามค่าที่ได้รับ
	if classNo==0 : #เป็น Normal เมื่อคืนค่าเป็น 0
		return "Normal"
	elif classNo==1:
		return "Risk of Brain Tumor" #เป็น Risk of Brain Tumor เมื่อคืนค่าเป็น 1


def getResult(img): #ฟังก์ชันที่ใช้ในการทำนายผลลัพธ์ของภาพ
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    return result
    #result=model.predict(input_img)
    #class_index = np.argmax(result)
    #return class_index
    


@app.route('/', methods=['GET']) #ประกาศเส้นทางสำหรับหน้าแรกของเว็บแอปพลิเคชัน
def index():
    return render_template('index.html') #ฟังก์ชันที่คืนเทมเพลตของหน้าแรก


@app.route('/predict', methods=['GET', 'POST']) #ประกาศเส้นทางสำหรับการทำนายผลลัพธ์
def upload(): #ฟังก์ชันที่ใช้ในการอัปโหลดภาพและทำนายผลลัพธ์
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
        #class_index =getResult(file_path)
        #class_name =get_className(class_index) 
        #return class_name
    return None


if __name__ == '__main__': #เช็คว่าโปรแกรมถูกเรียกโดยตรงหรือไม่
    app.run(debug=True) #เริ่มการทำงานของเซิร์ฟเวอร์ Flask บน localhost พร้อมการเปิดโหมด debug