# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:49:42 2019

@author: Vivekanandan | Techvantage
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:37:20 2019

@author: Vivekanandan | Techvantage
"""

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import  array_to_img, img_to_array, load_img
import pandas as pd
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from matplotlib import pyplot as plt
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import time
import imutils


size = (300, 300)
def transform_image(size):
    # function for transforming images into a format supported by CNN
    x = load_img('aa.jpg', target_size=(size[0], size[1]) )
    x = img_to_array(x) / 255
    x = np.expand_dims(x, axis=0)
    return (x)

def show_image(IMAGE):
    plt.figure(figsize=(10,10))
    plt.imshow(IMAGE, aspect = 'auto')
    plt.show()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path='aadhar_27_9_2019__model_tflite.tflite')
# interpreter = tf.lite.Interpreter(model_path='adhar_pan_license_17-10-19.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

CLASSES = ['name','dob','gender','no','front','address','back']
#CLASSES =['permit_no','kitas_front']
# CLASSES = ["cow", "cow_face", "unblurred_muzzle", "cow_eye", "cow_eye",
#     "cow_ear", "cow_ear", "cow_eartag", "laptop", "mobile", "cup", "chair",
#     "pen", "mouse", "monitor", "book", "bottle", "key_board",
#     "tablet", "person", "eye_glass","cpu","blurred_muzzle", "adhar_front", "adhar_back", "pan", "license"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

image = 'test2.jpg'
frame = cv2.imread(image)
(h, w) = frame.shape[:2]
cv2.imwrite('aa.jpg',frame)

# change the following line to feed into your own data.
img = transform_image(size)
#classifier.predict_classes(img)[0][0]
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

for index,detection  in Result_Data.iterrows():
    confidence = detection['score']

    if confidence > 10:
            idx = int(detection['class'])
            (startX, startY, endX, endY) = detection['x1'].astype("int"),detection['y1'].astype("int"),detection['x2'].astype("int"),detection['y2'].astype("int")
            crop=frame[startY:endY,startX:endX]
            if int(detection['class']) != 4:
                import pytesseract 
                text= pytesseract.image_to_string(crop)
                print(text)
            
            
            #cv2.imshow('crop',crop)
            #cv2.waitKey(0)
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],confidence)
            cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow("Frame", cv2.resize(frame,(800,800)))
key = cv2.waitKey(0)

cv2.destroyAllWindows()
