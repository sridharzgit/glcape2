# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:57:35 2020

@author: 100119
"""

import glob
import cv2
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os
count = 232
for image in glob.glob('C:/Users/100119/Desktop/kyc_retrain/pan_augmented/*.jpg'):
     print(image)
     count=count+1
     image=cv2.imread(image)   
#     image = cv2.resize(image,(300,300),fx=2.5,fy=2.5)
     image = cv2.resize(image,None,fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
     plt.imshow(image)
     plt.show()
     #Save Each Transformation
     cv2.imwrite('C:/Users/100119/Desktop/kyc_retrain/resized_images/00000_new_' +  str(count)+ '.jpg',image)
#     os.rename(image,'D:/DOC_EXTRACT/train/doc_res_' +str(count)+ '.jpg')
     count=count+1