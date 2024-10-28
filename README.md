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

The model is built using a simple feedforward neural network with the following architecture:

    - Input Layer: Flattening the 28x28 images into a 784-dimensional vector.
    - Hidden Layer 1: 128 neurons with ReLU activation.
    - Hidden Layer 2: 64 neurons with ReLU activation. 
    - Output Layer: 10 neurons with softmax activation (representing classes 0-9).


## Results

The model achieves an accuracy of over 98.23% on the test dataset. Training and validation accuracy and loss can be visualized using Matplotlib during the training process.

## Future Improvements

Experiment with more complex architectures (e.g., Convolutional Neural Networks).
Implement data augmentation techniques to improve model generalization.
Optimize hyperparameters for better performance.


