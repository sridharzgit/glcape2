# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:39:52 2019

@author: 100124
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(
        #rotation_range=2,
        width_shift_range=0.0,
        height_shift_range=0.0,
        shear_range=0.1,
        zoom_range=0.1
        #horizontal_flip=True,
#        fill_mode='nearest'
        )
directory = 'augment original adhar'
for filename in os.listdir(directory):
    img = load_img(directory+"\\"+filename)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    j = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='preview_3', save_prefix="image", save_format='jpg'):
        i += 1
        j += 1
        if i > 5:
            break  # otherwise the generator would loop indefinitely