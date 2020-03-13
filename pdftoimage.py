# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:11:32 2019

@author: 100119
"""

import os
#import tempfile
from pdf2image import convert_from_path

PDF_DIR = 'C:/Users/100119/Desktop/PRUDENTIAL/HANDWRITTEN_DATA_EXTRACTION/HANDWRITTEN_FILLED_FORM_PDF'
for pdf in os.listdir(PDF_DIR):

    filename = os.path.join(PDF_DIR,pdf)
    print(filename)
    pages = convert_from_path(filename, 500)
    base_filename  =  os.path.splitext(os.path.basename(filename))[0]
    save_dir = 'C:/Users/100119/Desktop/PRUDENTIAL/HANDWRITTEN_DATA_EXTRACTION/IMAGES_HANDWRITTEN'
    count = 1
    for page in pages:
        page.save(save_dir + '/' + base_filename + '_'+ str(count) + '.jpg')
        count += 1
