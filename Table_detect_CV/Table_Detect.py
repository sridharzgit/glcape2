# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:01:14 2019

@author: 100119
"""
import os
import cv2
import numpy as np
os.chdir('C:/Users/100119/Desktop/Table_detect_CV/output')
image_path = 'D:/DOC_EXTRACT/Dataset/600x1024/test/doc_res_1070.jpg'
img = cv2.imread(image_path,0)
basename =os.path.splitext(os.path.basename(image_path))[0]

(thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY|     cv2.THRESH_OTSU)
edges = cv2.Canny(img, 50, 150, apertureSize=3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength, maxLineGap)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite(basename + '_0001.jpg', img)
cv2.destroyAllWindows()
