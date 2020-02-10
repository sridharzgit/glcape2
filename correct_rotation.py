# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:08:36 2020

@author: 100119
"""
    
import numpy as np
import cv2
import math
from scipy import ndimage


img_before = cv2.imread('C:/Users/100119/Desktop/kyc_retrain/fire_base_images/212f.jpg')


img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)


angles = []


for x1, y1, x2, y2 in lines[0]:
    cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angles.append(angle)


median_angle = np.median(angles)
img_rotated = ndimage.rotate(img_before, median_angle)
print ("Angle is {}".format(median_angle))
cv2.imwrite("straighten.jpg",img_rotated)
#cv2.waitKey(0)
#cv2.destroyAllWindows()






