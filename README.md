# Handwritten Digits Classification

## Overview
This project aims to build a machine learning model to classify handwritten digits (0-9) using the MNIST dataset. The model leverages TensorFlow and Keras for deep learning, along with NumPy and Matplotlib for data manipulation and visualization.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Features
- Load and preprocess the MNIST dataset.
- Build a neural network model for digit classification.
- Train the model and evaluate its performance.
- Visualize sample images and their predicted labels.

## Technologies Used
- Python 3.17
- TensorFlow
- Keras
- NumPy
- Matplotlib

## Dataset
The MNIST dataset is a collection of 70,000 handwritten digits (60,000 training images and 10,000 test images). Each image is a 28x28 pixel grayscale image. The dataset is publicly available and can be downloaded from [here](http://yann.lecun.com/exdb/mnist/).

## Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:EmmaSHANG0625/MNIST-Handwriten-Digit-Classification.git

2. Install the required packages: 
    pip install -r requirements.txt

## Usage
1. Ensure the MNIST dataset files are placed in the dataset folder as follows:
    train-images.idx3-ubyte
    train-labels.idx1-ubyte
    t10k-images.idx3-ubyte
    t10k-labels.idx1-ubyte

2. Run the main script:
    python main.py

3. The model will train for a specified number of epochs, and the training history will be displayed.

4. After training, the model will be saved as mnist_model.h5.


## Model Architecture

The model is firstly built using a simple feedforward neural network with the following architecture:
- Input Layer: Flattening the 28x28 images into a 784-dimensional vector.
- Hidden Layer 1: 128 neurons with ReLU activation.
- Hidden Layer 2: 64 neurons with ReLU activation. 
Ò- Output Layer: 10 neurons with softmax activation (representing classes 0-9).

This project uses a Convolutional Neural Network (CNN) for the MNIST handwritten digit classification. CNNs are particularly effective for image recognition tasks due to their ability to detect patterns and spatial hierarchies in images. Below is an outline of the architecture used:

1. Input Layer:
    - Reshape Layer: Reshapes each input image from a 2D array (28x28) to a 3D array (28x28x1) to include a channel dimension required for CNNs.

2. Convolutional Layers:
    - Conv2D (32 filters, 3x3 kernel, ReLU activation, padding='same'): This layer applies 32 filters with a 3x3 kernel size, detecting edges and simple patterns within the image.
    - MaxPooling2D (2x2 pool size): Reduces the spatial dimensions of the feature maps, decreasing computation and helping prevent overfitting by summarizing local information.

3. Additional Convolutional Layers:
    -  Conv2D (64 filters, 3x3 kernel, ReLU activation, padding='same'): Learns more complex patterns with 64 filters, further enhancing the ability to capture details in the image.
    - MaxPooling2D (2x2 pool size): Further reduces the spatial dimensions, maintaining only the most important features.

4. Flatten Layer:
    - Converts the 3D feature maps output by the convolutional layers into a 1D vector to prepare for the fully connected layers.

5. Fully Connected (Dense) Layers:
    - Dense (128 units, ReLU activation): A dense layer with 128 units learns complex representations of the flattened features.
    - Dense (10 units, softmax activation): The final output layer has 10 units, each representing a class (digits 0–9) with a softmax activation function to produce probability scores for each digit.
    
6. Compilation:
    - The model is compiled using the Adam optimizer, with categorical cross-entropy as the loss function (suitable for multi-class classification) and accuracy as the evaluation metric.

### Summary of Architecture:

| Layer                  | Output Shape      | Parameters |
|------------------------|-------------------|------------|
| Reshape                | (28, 28, 1)      | 0          |
| Conv2D (32 filters)    | (28, 28, 32)     | 320        |
| MaxPooling2D           | (14, 14, 32)     | 0          |
| Conv2D (64 filters)    | (14, 14, 64)     | 18,496     |
| MaxPooling2D           | (7, 7, 64)       | 0          |
| Flatten                | (3136)           | 0          |
| Dense (128 units)      | (128)            | 401,536    |
| Dense (10 units)       | (10)             | 1,290      |

**Total Parameters**: 421,642


This CNN architecture allows the model to progressively learn and capture intricate details in the handwritten digits, ultimately providing robust classification performance for digit recognition tasks.


## Results

The model achieves an accuracy of over 98.23% for the first model, and 99.07% for the CNN model, on the test dataset. Training and validation accuracy and loss can be visualized using Matplotlib during the training process.

## Future Improvements

Implement data augmentation techniques to improve model generalization.
Optimize hyperparameters for better performance.


