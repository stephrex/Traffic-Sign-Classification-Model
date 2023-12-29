import tensorflow as tf
from skimage import transform
from skimage import exposure
from skimage import io
import argparse
import os
import random
import numpy as np
import cv2
from imutils import paths
import imutils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='path to saved model')
ap.add_argument('-i', '--images', required=True,
                help='Path to dir containing testing images')
ap.add_argument('-p', '--predictions', required=True,
                help='Path to dir to save predictions from model')
args = vars(ap.parse_args())

# load the model from disk
model_path = args['model']
model = tf.keras.models.load_model(model_path)

# Load sample images from disk
labelNames = open('signnames.csv').read().strip().split('\n')[1:]
labelNames = [l.split(',')[1] for l in labelNames]

# Grab the paths to the input images, shuffle them, and grab a sample
print('[INFO] Predicting...')
imagePaths = list(paths.list_images(args['images']))
random.shuffle(imagePaths)
imagePaths = imagePaths

# Loop over the image paths to preprocess and make predictions
for (i, imagePath) in enumerate(imagePaths):
    # get the image, convert to numbers, resize it, and equalize the contrast
    image = io.imread(imagePath)
    image = transform.resize(image, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1, nbins=288)
    # Scale the image
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    # make predictions using the saved traffic sign classififcation CNN model
    preds = model.predict(image)
    j = preds.argmax(axis=1)[0]
    pred_label = labelNames[j]
    # load the image using opencv, resize and view the predicited label
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=244)
    cv2.putText(image, pred_label, (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # save the image to disk
    pred_path = os.path.sep.join([args['predictions'], '{}.png'.format(i)])
    cv2.imwrite(pred_path, image)

# python predict.py --model C:/Users/User/Documents/Machine/Object-Detection-Project-TSF-Internship/Traffic-Sign-Classification-Project(Github)/Output/trafficSignCNN_10.model --images C:/Users/User/Documents/Machine/Object-Detection-Project-TSF-Internship/Traffic-Sign-Classification-Project(Github)/GTSRB-Dataset/Test --predictions C:/Users/User/Documents/Machine/Object-Detection-Project-TSF-Internship/Traffic-Sign-Classification-Project(Github)/Predicted_images/testing
