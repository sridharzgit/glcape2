# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:46:53 2020

@author: Nayana
"""
#import pyautogui
import pytesseract
import pandas as pd
import re

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:28:08 2020

@author: 100119
"""
import os
#import ocrmypdf
#import pytesseract
import errno
import shutil
import cv2
import warnings  
import numpy as np
from PIL import Image
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
#Root DIRECTORY
WORKING_DIR = os.getcwd()
EXTRACTION_DPI = 300
MAX_NUM_BOXES = 5
MIN_SCORE = 0.5
INFERENCE_GRAPH = WORKING_DIR+'/common/Modeling_Code/models/frozen_inference_graph.pb'
PATH_TO_EXTRACTED_IMAGES = None
#PDF_PATH ='C:/Users/100119/Desktop/table_train/Table_OCR/pdf/Latex_100491-converted.pdf'
TABLE_FOLDER = WORKING_DIR+'/common/Modeling_Code/cropped_images'



def do_inference_with_graph(PATH_TO_IMAGE, inference_graph_path):
    
    detection_graph = tf.Graph()
    # checking if inference graph exists
    if not os.path.isfile(inference_graph_path):
        print('Inference graph at\n{}\nnot found'.format(inference_graph_path))

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image = cv2.imread(PATH_TO_IMAGE)
            image_expanded = np.expand_dims(image, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_expanded})
            return boxes[0], scores[0], classes[0]

def check_if_intersected(coord_a, coord_b):
    
    return \
        coord_a['x_max'] > coord_b['x_min'] and \
        coord_a['x_min'] < coord_b['x_max'] and \
        coord_a['y_max'] > coord_b['y_min'] and \
        coord_a['y_min'] < coord_b['x_max']

def check_if_vertically_overlapped(box_a, box_b):
  
    return \
        box_a['y_min'] < box_b['y_min'] < box_a['y_max'] or \
        box_a['y_min'] < box_b['y_max'] < box_a['y_max'] or \
        (box_a['y_min'] >= box_b['y_min'] and box_a['y_max'] <= box_b['y_max']) or \
        (box_a['y_min'] <= box_b['y_min'] and box_a['y_max'] >= box_b['y_max'])

def merge_vertically_overlapping_boxes(boxes):
    
    # first box is always inside
    merged_boxes = [boxes[0]]
    i = 0
    overlapping = False
    for box in boxes[1:]:
        i += 1
        # extraction of coordinates for better reading
        coord_box = {
            'y_min': box[0],
            'x_min': box[1],
            'y_max': box[2],
            'x_max': box[3]
        }
        for m_box in merged_boxes:
            # extraction of coordinates for better reading
            coord_m_box = {
                'y_min': m_box[0],
                'x_min': m_box[1],
                'y_max': m_box[2],
                'x_max': m_box[3]
            }

            if check_if_vertically_overlapped(coord_m_box, coord_box):
                overlapping = True
                # merge of the two overlapping boxes
                if m_box[0] > box[0]:
                    m_box[0] = box[0]
                if m_box[2] < box[2]:
                    m_box[2] = box[2]
        if not overlapping:
            # if not overlapping we append the box. Exit condition for recursive call
            merged_boxes.append(box)
    if overlapping:
        # recursive call. It converges because the exit condition consumes the generator.
        return merge_vertically_overlapping_boxes(merged_boxes)
    else:
        return merged_boxes

def keep_best_boxes_merged(boxes, scores, classes, max_num_boxes=MAX_NUM_BOXES,  min_score=0.5):
    
    kept_scores = []
    kept_boxes = [] 
    kept_classes = [] # always keep the firs box, which is the best one.
    num_boxes = 0
    i = 0
    print("score",scores[0])
    if scores[0] > min_score:
#        ("scores", scores[0])
        kept_boxes.append(boxes[0])
        kept_scores.append(scores[0])
        kept_classes.append(classes[0])
        num_boxes += 1
        i += 1
        for b in boxes[1:]:
            # add boxes to the ones to be merged
            if num_boxes < max_num_boxes and scores[i] > min_score:

                kept_boxes.append(b)
                num_boxes += 1
                kept_scores.append(scores[i])
                kept_classes.append(classes[i])

                i += 1
            else:
                break
        print("k", kept_boxes)
        kept_boxes = merge_vertically_overlapping_boxes(kept_boxes)
    else:
        kept_boxes = []
    print("kept_boxes",kept_boxes)
    return kept_boxes, kept_scores, kept_classes

def crop_wide(pil_image, boxes):
    
    cropped_tables = []
    segments = [0]  # adding position 0 to simplify anti-crop text later
    height_of_crops = 0
    (im_width, im_height) = pil_image.size
#    (0, int(box[0]), im_width, int(box[2]))))
#    (int(box[1]), int(box[0]), int(box[3]), int(box[2]))
    cropped_tables =[pil_image.crop(tuple((int(box[1]), int(box[0]), int(box[3]), int(box[2])))) for box in boxes if not boxes == []]
    if not boxes == []:
        
        
        for box in boxes:
#            (0, int(box[0]), im_width, int(box[2]))))
#            cropped_tables.append(pil_image.crop(tuple((int(box[1]), int(box[0]), int(box[3]), int(box[2])))))
            segments.append(int(box[0]))
            segments.append(int(box[2]))
            height_of_crops += (int(box[2]) - int(box[0]))
        # sorts all segments to simplify anti-crop text later
        segments.append(im_height)  # adding last position to simplify anti-crop text later
        segments.sort()

        # create new image with new dimension
        new_image = Image.new('L', (im_width, im_height - height_of_crops))
        start_position = 0
        # cutting image in anti-boxes position
        for i in range(len(segments)):  # segments will always be even
            if not i % 2 and i < len(segments) - 1:  # takes only even positions
                if i != 0:
                    start_position += segments[i - 1] - segments[i - 2]
                new_image.paste(pil_image.crop(tuple((0, segments[i], im_width, segments[i + 1]))), (0, start_position))
        cropped_text = new_image
        print('Created text image')

    else:
        print('No boxes found')
        cropped_text = pil_image

    return cropped_tables, cropped_text

def extract_tables_and_text(PATH_TO_IMAGE, inference_graph_path):
    image = cv2.imread(PATH_TO_IMAGE)
    (im_height, im_width, _) = image.shape
    boxes, scores, classes = do_inference_with_graph(PATH_TO_IMAGE, inference_graph_path)
#    print(boxes, scores)
    best_boxes, best_scores, best_classes = keep_best_boxes_merged(
        boxes=boxes,
        scores=scores,
        classes = classes,
        max_num_boxes=MAX_NUM_BOXES,
        min_score=MIN_SCORE
    )
    # create coordinates based on image dimension
    for box in best_boxes:
        box[0] = int(box[0] * im_height)
        box[2] = int(box[2] * im_height)
        box[1] = int(box[1] * im_width)
        box[3] = int(box[3] * im_width)
    pil_image = Image.open(PATH_TO_IMAGE)
    (cropped_tables,_) = crop_wide(pil_image, best_boxes)
    return cropped_tables, best_classes

def Extract_Table(file_name,cropped_tables, temp_table_path, classes):
    file_name = file_name.split("/")[-1]
    file_name = file_name.split(".")[0]
    i = 0
    table_paths = []
    print(classes)
    if cropped_tables is not None:
        
        for ct in cropped_tables:
            if classes[i]==1.0:
                class1= "table"
            elif classes[i]==2.0:
                class1 = "no"
            elif classes[i]==3.0:
                class1 = "text"
            new_file_path = \
                os.path.join(temp_table_path,
                             file_name+class1+'.jpg')
            print(new_file_path)
#            ct = ct.convert('L')
#            sd = deskew.Deskew(
#                input_numpy=np.asarray(ct),
#                output_numpy=True
#            )
#            de_skewed_image_np = sd.run()
#            ct = Image.fromarray(de_skewed_image_np)
#            ct = ct.convert(mode='L')
            try:
                ct.save(new_file_path, dpi=(EXTRACTION_DPI, EXTRACTION_DPI))
            except IOError or ValueError as e:
                print('Cannot write image on disk: \n{}'.format(e))
            i += 1
            table_paths.append(new_file_path)
            
        print('Writing cropped tables done.')
    else:
        print('No tables to write on disk')
    return table_paths


def detect_table(image):
    c_tables, classes = extract_tables_and_text(image,INFERENCE_GRAPH)
    table_paths = Extract_Table(file_name = image,
            cropped_tables = c_tables,
            temp_table_path=TABLE_FOLDER,
            classes=classes
            )
