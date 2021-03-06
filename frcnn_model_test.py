# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:27:03 2019

@author: 100119
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
#os.chdir('C:/Users/100119/Desktop/table_train')
# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


PATH_TO_CKPT = 'C:/Users/100119/Desktop/PRUDENTIAL/frozen_inference_graph.pb'
#'C:/Users/100119/Desktop/table_train/model__rcnn_inception_adam_4/frozen_inference_graph.pb'
#'C:/Users/100119/Desktop/table_train/mymodel/frozen_inference_graph.pb'
PATH_TO_LABELS = 'C:/Users/100119/Desktop/PRUDENTIAL/objectdetection_1.pbtxt'
PATH_TO_IMAGE ='C:/Users/100119/Desktop/PRUDENTIAL/dataset/data/Invoice-B-0191-Playful-page-001.jpg'
NUM_CLASSES = 7

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection (aka 'visulaize the results')

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=2,
    min_score_thresh=0.60)


width, height = image.shape[:2]

cv2.imshow('Object detector', cv2.resize(image,(1000,800),fx=2.5,fy=2.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
