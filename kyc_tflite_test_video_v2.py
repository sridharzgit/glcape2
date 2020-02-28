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


size = (300,300)
def transform_image(frame, size):
    # function for transforming images into a format supported by CNN
    x = load_img('aa.jpg', target_size=(size[0], size[1]) )
    x = img_to_array(x) / 255
    x = np.expand_dims(x, axis=0)
    return (x)

def show_image(IMAGE):
    plt.figure(figsize=(10,10))
    plt.imshow(IMAGE, aspect = 'auto')
    plt.show()


#image_path = 'E:/PROJECTS/Cattle/00-cow/153.jpg'
#image = cv2.imread(image_path )
#image = cv2.resize(image,(100,100),fx=2.5,fy=2.5)

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path='E:/Desktop/KYC_EXTRACT/Version2/model_v2/kyc_detection_model_v2__30_10_2019_tflite.tflite')
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

#CLASSES = ["cow", "cow_face", "cow_muzzle", "cow_L_eye", "cow_R_eye","cow_L_ear", "cow_R_ear",
#	 "cow_eartag","cow_left_side","cow_right_side", "laptop", "mobile","cup","chair","pen","mouse",
#	"monitor","book","bottle", "keyboard","tablet","person", "eye_glass","cpu","cracked_screen"]
CLASSES = ["cow", "cow_face", "unblurred_muzzle", "cow_eye", "cow_eye",
     "cow_ear", "cow_ear", "cow_eartag", "laptop", "mobile", "cup", "chair",
     "pen", "mouse", "monitor", "book", "bottle", "key_board",
     "tablet", "person", "eye_glass","cpu","blurred_muzzle", "adhar_front", "adhar_back", "pan", "license","atm","business_card","office_id"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
	  # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame,width=1000,height=1000)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    cv2.imwrite('aa.jpg',frame)

    # change the following line to feed into your own data.
    img = transform_image(frame, size)
    #classifier.predict_classes(img)[0][0]
    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    Result_Data = pd.DataFrame(output_data[0],columns=['y1','x1','y2','x2'])
    output_data = interpreter.get_tensor(output_details[1]['index'])
    Result_Data['class'] = list(output_data[0])
    output_data = interpreter.get_tensor(output_details[2]['index'])
    Result_Data['score'] = list(output_data[0])
    Result_Data['x1'] = (Result_Data['x1']*w).astype(int)
    Result_Data['y1'] = (Result_Data['y1']*h).astype(int)
    Result_Data['x2'] = (Result_Data['x2']*w).astype(int)
    Result_Data['y2'] = (Result_Data['y2']*h).astype(int)

    for index,detection  in Result_Data.iterrows():
        confidence = detection['score']*100

        if confidence > 10:
                idx = int(detection['class'])
                (startX, startY, endX, endY) = detection['x1'].astype("int"),detection['y1'].astype("int"),detection['x2'].astype("int"),detection['y2'].astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],confidence)
                cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cv2.destroyAllWindows()

