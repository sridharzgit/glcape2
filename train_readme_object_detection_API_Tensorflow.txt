REFFERENCE : https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
NOTE: always run cmd as admin mode
step 1:
open cmd

step 2:
activate virtual environment (tensorflow1)

step 3:
Configure PYTHONPATH environment variable
(set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim)

step 4:
goto object detection folder
cd C:\tensorflow1\models\research\object_detection

step 5:
copy paste the train and test data to C:\tensorflow1\models\research\object_detection\images folder.

Also, you can check if the size of each bounding box is correct by running sizeChecker.py (python sizeChecker.py --move)

step 6:
Generate Training Data (python xml_to_csv.py)

step 7:
edit generate_tfrecord.py:
Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number.

step 8:
Then, generate the TFRecord files by issuing these commands from the \object_detection folder:

python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record

step 9:
Use a text editor to edit labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\training folder. 

step 10:
Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and 
copy the faster_rcnn_inception_v2_pets.config file into the \object_detection\training directory. 
Then, open the file with a text editor. There are several changes to make to the .config file, mainly 
changing the number of classes and examples, and adding the file paths to the training data.

step 11:
Make the following changes to the faster_rcnn_inception_v2_pets.config file. 

Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path 
error when trying to train the model! 
Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).

	Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above 
		basketball, shirt, and shoe detector, it would be num_classes : 3 .

	Line 106. Change fine_tune_checkpoint to:

		fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

	Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:
		input_path : "C:/tensorflow1/models/research/object_detection/train.record"
		label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

	Line 132. Change num_examples to the number of images you have in the \images\test directory.

	Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:

		input_path : "C:/tensorflow1/models/research/object_detection/test.record"
		label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

step 12:
Run the Training

python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config

if  using ssd model


python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_coco.config


I recommend allowing your model to train until the loss consistently drops below 0.05, which will take about 40,000 steps,
or about 2 hours (depending on how powerful your CPU and GPU are). Note: The loss numbers will be different if a different
model is used. MobileNet-SSD starts with a loss of about 20, and should be trained until the loss is consistently under 2.


You can view the progress of the training job by using TensorBoard. 
To do this, open a new instance of Anaconda Prompt, activate the tensorflow1 virtual environment, 
change to the C:\tensorflow1\models\research\object_detection directory, and issue the following command
to view graf (tensorboard --logdir=training)

then open http://localhost:6006/ in any browser.

step 13:
Export Inference Graph
Now that training is complete, the last step is to generate the frozen inference graph (.pb file). 
From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with 
the highest-numbered .ckpt file in the training folder:

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

python export_tflite_ssd_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

This creates a frozen_inference_graph.pb file in the \object_detection\inference_graph folder. The .pb file contains the 
object detection classifier.

Evaluate Model:
======================
python eval.py --logtostderr --checkpoint_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v2_coco.config  --eval_dir=eval/
#tensorboard
tensorboard --logdir=eval/
