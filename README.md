# Traffic-Sign-Classification-Model

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras for the classification of traffic signs and symbols. The model is trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

## Getting Started

### Prerequisites

Make sure you have the following dependencies installed:

- Python (3.6 or later)
- TensorFlow
- scikit-image
- scikit-learn
- matplotlib
- imutils

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Dataset
Download the GTSRB dataset, and organize it as follows.
```
- GTSRB-Dataset
  - Train.csv
  - Test.csv
  - Train
    - 0
      - 00000_00000.png
      - ...
    - ...
  - Test
    - 0
      - 00000_00000.png
      - ...
    - ...
```

## Usage

### Train the Model

To train the traffic sign recognition model, use the following command:

```bash
python train_model.py --dataset /path/to/GTSRB-Dataset --model /path/to/output/model.model --plot /path/to/training/output/history/plot.png
```

This command will load and preprocess the dataset, train the model, and save both the trained model and a plot of the training history.

### Make Predictions

To make predictions on a set of testing images, use the following command:

```bash
python predict_model.py --model /path/to/saved/model.h5 --images /path/to/testing/images --predictions /path/to/save/predictions
```

This command will load the trained model and make predictions on the specified testing images, saving the result images with predicted labels.

## Model Architecture
The CNN model architecture is defined in the `Traffic_Sign_CNN` class in `Model_Network/Traffic_sign_recognition_CNN.py`. It consists of Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, and Dropout layers.

Feel free to experiment with the architecture or hyperparameters to further improve the model's performance.

## Results
During training, the model's performance is evaluated on the testing set in the dataset, and the classification report is printed.

The validation loss and accuracy during training are available in the saved training history plot (`plot.png`). Additionally, the model's predictions on testing images are saved to the predictions directory.


```bash 
Make sure to adjust the paths according to your actual project structure.
```

