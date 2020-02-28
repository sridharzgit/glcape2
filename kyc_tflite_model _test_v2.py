# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:49:42 2019

@author: Vivekanandan | Techvantage
"""
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
import pandas as pd
from matplotlib import pyplot as plt
import cv2



SIZE = (300, 300)
IMAGE_PATH ='C:/Users/100119/Desktop/kyc_retrain/fire_base_images/aadhar/Aadhar - I.jpg'
MODEL_PATH ='C:/Users/100119/Desktop/kyc_retrain/kyc_model_07_02_2020/kyc_model_tflite_07_02_2020.tflite'
def transform_image(size):
    # function for transforming images into a format supported by CNN
    x = load_img('aa.jpg', target_size=(size[0], size[1]) )
    x = img_to_array(x) / 255
    x = np.expand_dims(x, axis=0)
    return (x)

def show_image(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image, aspect = 'auto')
    plt.show()

def model_test(image_path,model_path,size):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    # interpreter = tf.lite.Interpreter(model_path='adhar_pan_license_17-10-19.tflite')
    interpreter.allocate_tensors()
    
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    
    #CLASSES = ['name','dob','gender','no','front','address','back']
    #CLASSES =['permit_no','kitas_front']
    
    CLASSES = ["cow_eartag", "laptop", "mobile", "cup", "chair",
         "pen", "mouse", "monitor", "book", "bottle", "key_board",
         "tablet", "person", "eye_glass","cpu", "adhar_front", "adhar_back", "pan", "license","atm","business_card","office_id","paper","adhar_no"]
    
    #CLASSES =['pan','pan_no','pan_name','pan_dob']
    
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    
    frame = cv2.imread(image_path)
    (h, w) = frame.shape[:2]
    cv2.imwrite('aa.jpg',frame)
    img = transform_image(size)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    Result_Data = pd.DataFrame(output_data[0],columns=['y1','x1','y2','x2'])
    output_data = interpreter.get_tensor(output_details[1]['index'])
    Result_Data['class'] = list(output_data[0])
    output_data = interpreter.get_tensor(output_details[2]['index'])
    Result_Data['score'] = list(output_data[0])
    Result_Data['score'] = Result_Data['score'] * 100
    Result_Data['x1'] = (Result_Data['x1']*w).astype(int)
    Result_Data['y1'] = (Result_Data['y1']*h).astype(int)
    Result_Data['x2'] = (Result_Data['x2']*w).astype(int)
    Result_Data['y2'] = (Result_Data['y2']*h).astype(int)
    
    count =1
    for index,detection  in Result_Data.iterrows():
        confidence = int(detection['score']) 
        if confidence >10:
                idx = int(detection['class'])
                (startX, startY, endX, endY) = detection['x1'].astype("int"),detection['y1'].astype("int"),detection['x2'].astype("int"),detection['y2'].astype("int")
                crop=frame[startY:endY,startX:endX]
                cv2.imwrite('C:/Users/100119/Desktop/kyc_retrain/fire_base_images/' +str(count)+'.jpg',crop)
               
                print(CLASSES[idx],"\t Score : ",confidence)
    #            if CLASSES[idx] == 'adhar_no':
    #                cv2.rectangle(frame, (startX, startY), (endX, endY),(0,0,255), -1)
                 
                count+=1
    
                cv2.imshow('crop',crop)
                cv2.waitKey(0)
    
                label = "{}: {:.2f}%".format(CLASSES[idx],confidence)
                cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 3)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
        else:
            print("Not Found Any Object")
    
    cv2.imshow("Frame", cv2.resize(frame,(400,400)))
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
   
model_test(IMAGE_PATH,MODEL_PATH,SIZE)