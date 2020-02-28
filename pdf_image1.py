# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:22:13 2019

@author: 100119
"""
import os
import time
import uuid
#import tempfile
from pdf2image import convert_from_path
start = time.time()
PDF_DIR = 'C:/Users/100119/Desktop/PRUDENTIAL/CREDIT_CARD_NO_MASKING/data/Forms'
save_dir ='C:/Users/100119/Desktop/PRUDENTIAL/CREDIT_CARD_NO_MASKING/data/images'
for pdf in os.listdir(PDF_DIR):

    filename = os.path.join(PDF_DIR,pdf)
    print(filename)
    pages = convert_from_path(filename, dpi=200,fmt="jpg", output_file=str(uuid.uuid4()),\
                              output_folder=save_dir,thread_count=2)
    # base_filename  =  os.path.splitext(os.path.basename(filename))[0]
    # count = 1
    # for page in pages:
    #     page.save(save_dir + '/' + base_filename + '_'+ str(count) + '.jpg')
    #     count += 1
print("Time:\t", round(time.time()-start,3),"\nDone")