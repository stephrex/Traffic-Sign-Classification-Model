# import necessary packages
import matplotlib.pyplot as plt
import argparse
import os
import random
import numpy as np
import tensorflow as tf
from skimage import io
from skimage import exposure
from skimage import transform
from sklearn.metrics import classification_report
from Model_Network.Traffic_sign_recognition_CNN import Traffic_Sign_CNN


# Set matplotlib to save figure in background
import matplotlib
matplotlib.use('Agg')


def load_split(basePath, csvPath):
    data = []
    labels = []

    # extract the rows from the csv file
    rows = open(csvPath).read().strip().split('\n')[1:]
    # print(rows)

    # shuffle the rows
    random.shuffle(rows)

    for (i, row) in enumerate(rows):
        # check status update for each 1000th prepreprocessed images
        if i > 0 and i % 1000 == 0:
            print(f"[INFO] processed {i} total images")

        # split the rows and grab the classid and imagepath
        (label, imagepath) = row.strip().split(',')[-2:]
        # print(f'label: {label}, imagepath: {imagepath}')

        # get the image path
        imagepath = os.path.sep.join([basePath, imagepath])
        # convert image to numbers
        image = io.imread(imagepath)
        # resize the image
        image = transform.resize(image, (32, 32))
        # increase image contrast
        image = exposure.equalize_adapthist(image, clip_limit=0.1, nbins=288)

        data.append(image)
        labels.append(int(label))

    # convert data and labels to numpy array
    data = np.array(data)
    labels = np.array(labels)

    # return a tuple of data and labels
    return (data, labels)

# view an exmple of what is returned by the load_split_function
# (data, labels) = load_split('C:/Users/User/Documents/Machine/Object-Detection-Project-TSF-Internship/Traffic_Sign_Classification_Project/GTSRB-Dataset',
#                             'C:/Users/User/Documents/Machine/Object-Detection-Project-TSF-Internship/Traffic_Sign_Classification_Project/GTSRB-Dataset/Train.csv')

# print(f'data: {data}, labels: {labels}')


# Parse the Command Line Arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to GTSRB Train Dataset')
ap.add_argument('-m', '--model', required=True, help='Path to output model')
ap.add_argument('-p', '--plot', type=str, default='plot.png',
                required=True, help='Path to training history plot')
args = vars(ap.parse_args())

# get the label names from the signames csv file
label_names = open('signnames.csv').read().strip().split('\n')[1:]
label_names = [i.split(',')[1] for i in label_names]

# Derive the paths to the Train images, and test images
Train_path_CSV = os.path.sep.join([args['dataset'], 'Train.csv'])
Test_path_CSV = os.path.sep.join([args['dataset'], 'Test.csv'])

# Extract the images and labels from the csv file, using the load_split function
(train_X, train_y) = load_split(args['dataset'], Train_path_CSV)
(test_X, test_y) = load_split(args['dataset'], Test_path_CSV)

# one hot encode the labels using tensorflow to categorical function
num_Labels = len(np.unique(train_y))
train_y = tf.keras.utils.to_categorical(train_y, num_Labels)
test_y = tf.keras.utils.to_categorical(test_y, num_Labels)

# Normalize the input images
train_X = train_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0

# Account for the class imabalance
classTotals = train_y.sum(axis=0)
classWeight = dict()
for i in range(len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]
print(classWeight)

# Augment the training data
aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10,
                                                      zoom_range=0.15,
                                                      width_shift_range=0.1,
                                                      height_shift_range=0.1,
                                                      shear_range=0.15,
                                                      fill_mode='nearest')

epochsNum = 7

# Compile the model
print('[INFO] compiling model.....')
model = Traffic_Sign_CNN.build(
    width=32, height=32, depth=3, classes=num_Labels)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3/(epochsNum * 0.5)),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
# Train the model
History = model.fit(aug.flow(x=train_X, y=train_y, batch_size=32),
                    validation_data=(test_X, test_y),
                    steps_per_epoch=train_X.shape[0] // 32,
                    epochs=epochsNum,
                    class_weight=classWeight,
                    verbose=2,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)])

# Evaluate the model
print('[INFO] evaluating model')
predictions = model.predict(test_X, batch_size=32)
print(classification_report(test_y.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=label_names))

# Save the model to disk
print('[INFO] Saving model to disk')
model.save(args['model'])

# plot the history
print('[INFO] Saving plot to disk')
num_epochs = np.arange(0, len(History.epoch))
plt.style.use('ggplot')
plt.figure()
plt.plot(num_epochs, History.history['loss'], label='train_loss')
plt.plot(num_epochs, History.history['accuracy'], label='train_accuracy')
plt.plot(num_epochs, History.history['val_loss'], label='val_loss')
plt.plot(num_epochs, History.history['val_accuracy'], label='accuracy')
plt.title('Training and Validation Accuracy and Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args['plot'], format='png')

# python train.py --dataset C:/Users/User/Documents/Machine/Object-Detection-Project-TSF-Internship/Traffic_Sign_Classification_Project/GTSRB-Dataset --model Output/trafficSignCNN_10.model --plot Output/model_10_plot.png
