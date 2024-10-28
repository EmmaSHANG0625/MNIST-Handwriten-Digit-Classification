import tensorflow as tf
import numpy as np
import os
import struct
import matplotlib.pyplot as plt

# Set up the dataset path
dataset_path = '/Users/emma/Desktop/becode_projects/MNIST-Handwriten-Digit-Classification/dataset'

# Function to read IDX files
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

def load_data():
    X_train = read_idx(os.path.join(dataset_path, 'train-images.idx3-ubyte'))
    y_train = read_idx(os.path.join(dataset_path, 'train-labels.idx1-ubyte'))
    X_val = read_idx(os.path.join(dataset_path, 't10k-images.idx3-ubyte'))
    y_val = read_idx(os.path.join(dataset_path, 't10k-labels.idx1-ubyte'))

    X_train, X_val = X_train / 255.0, X_val / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)

    return X_train, y_train, X_val, y_val

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)), 
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_val, y_val))
    score = model.evaluate(X_val, y_val, batch_size=128)
    print("Validation Loss and Accuracy: ", score)
    return history

def main():
    X_train, y_train, X_val, y_val = load_data()
    model = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)
    model.save('mnist_model.h5')
    print("Model training complete and saved as 'mnist_model.h5'.")

if __name__ == '__main__':
    main()
