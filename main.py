
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import struct
from tensorflow.keras.models import load_model

# Set up the dataset path
dataset_path = '/Users/emma/Desktop/becode_projects/MNIST-Handwriten-Digit-Classification/dataset'

# Check if the dataset folder and files exist
print("Files in dataset folder:", os.listdir(dataset_path))

# Define paths for each MNIST file
train_images_path = os.path.join(dataset_path, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(dataset_path, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(dataset_path, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(dataset_path, 't10k-labels.idx1-ubyte')

# Function to read IDX files
def read_idx(filename):
    """Reads an IDX file and returns the data as a NumPy array."""
    with open(filename, 'rb') as f:
        # Read magic number and dimensions
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        
        # Read the data
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

# 1. Define Functions for Each Step

def load_data():
    # Load and preprocess your data using the correct paths
    X_train = read_idx(train_images_path)
    y_train = read_idx(train_labels_path)
    X_val = read_idx(test_images_path)
    y_val = read_idx(test_labels_path)
    
    # Normalize and preprocess
    X_train, X_val = X_train / 255.0, X_val / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    
    return X_train, y_train, X_val, y_val

def build_model():
    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    # Train the model
    history = model.fit(X_train, y_train, epochs=25, batch_size=128, validation_data=(X_val, y_val))

    score = model.evaluate(X_val, y_val, batch_size=128)
    print("Validation Loss and Accuracy: ", score)
    return history

# 2. Define the Main Function

def main():
    # Load data
    X_train, y_train, X_val, y_val = load_data()
    
    # Build model
    model = build_model()
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Save model
    model.save('mnist_model.h5')
    print("Model training complete and saved as 'mnist_model.h5'.")

# 3. Run the Main Function When Script is Executed

if __name__ == '__main__':
    main()



