    # -*- coding: utf-8 -*-
    """
    Created on Mon Feb  3 14:39:14 2020
    
    @author: 100119
    """
    
    from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    import os
    
    datagen = ImageDataGenerator(
            rotation_range=2,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.3,
            zoom_range=0.3
            #horizontal_flip=True,
    #        fill_mode='nearest'
            )
    directory = 'C:/Users/100119/Desktop/kyc_retrain/pan_data_extraction/pan_data_extraction_data_06_02_2020/New folder'
    for filename in os.listdir(directory):
        img = load_img(directory+"/"+filename)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        
        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
        i = 0
        j = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='C:/Users/100119/Desktop/kyc_retrain/pan_data_extraction/pan_data_extraction_data_06_02_2020/aug_images', save_prefix="image_pan", save_format='jpg'):
            i += 1
            j += 1
            if i > 2:
                break  # otherwise the generator would loop indefinitely