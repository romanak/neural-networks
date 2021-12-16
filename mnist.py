#!/usr/bin/python3

# -*- coding: utf-8 -*-
"""
Created on Mon May 26 2021

A module to load MNIST data.

@author: Roman Akchurin
"""

import gzip
import numpy as np

def load_images(file_name):
    """Returns a Numpy array of the images for MNIST data set."""
    with gzip.open(file_name, 'r') as f:
        # first 4 bytes is a magic number
        # MSB is at the beginning of the byte array
        magic_number = int.from_bytes(f.read(4), byteorder='big', signed=False)
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), byteorder='big', signed=False)
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), byteorder='big', signed=False)
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), byteorder='big', signed=False)
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images

def load_labels(file_name):
    """Returns a Numpy array of the labels for MNIST data set."""
    with gzip.open(file_name, 'r') as f:
        # first 4 bytes is a magic number
        # MSB is at the beginning of the byte array
        magic_number = int.from_bytes(f.read(4), byteorder='big', signed=False)
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), byteorder='big', signed=False)
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels
    
def load_train_data():
    """Returns the training dataset with specified filenames."""
    train_images = load_images('data/train-images-idx3-ubyte.gz')
    train_labels = load_labels('data/train-labels-idx1-ubyte.gz')
    return (train_images, train_labels)

def load_test_data():
    """Returns the test dataset with specified filenames."""
    test_images = load_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = load_labels('data/t10k-labels-idx1-ubyte.gz')
    return (test_images, test_labels)

def load_data():
    """Helper function to load MNIST dataset."""
    return (load_train_data(), load_test_data())

def vectorize(labels):
    """Return a 2-dimensional tensor with the second vector
    consisting of 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    result = np.zeros((len(labels), 10, 1), dtype=np.uint8)
    for i in range(len(labels)):
        j = labels[i]
        result[i, j] = 1
    return result

def load_processed_data():
    """Returns preprocessed MNIST dataset for the neural network."""
    (train_images, train_labels) = load_train_data()
    (test_images, test_labels) = load_test_data()
    train_images = train_images.reshape((60000, 28*28, 1))
    train_images = train_images.astype('float32') / 255
    train_labels = vectorize(train_labels)
    train_data = zip(train_images[:49999], train_labels[:49999])
    train_data = list(train_data)
    valid_data = zip(train_images[50000:], train_labels[50000:])
    valid_data = list(valid_data)
    test_images = test_images.reshape((10000, 28*28, 1))
    test_images = test_images.astype('float32') / 255
    test_data = zip(test_images, test_labels)
    test_data = list(test_data)
    return (train_data, valid_data, test_data)

def load_matrix_data():
    """Returns dataset in a matrix-based form."""
    (train_images, train_labels) = load_train_data()
    (test_images, test_labels) = load_test_data()
    train_images = train_images.reshape((60000, 28*28))
    train_images = train_images.astype('float32') / 255
    train_images = np.asmatrix(train_images)
    train_images = train_images.transpose()
    train_labels = vectorize(train_labels)
    train_labels = np.asmatrix(train_labels)
    train_labels = train_labels.transpose()
    test_images = test_images.reshape((10000, 28*28))
    test_images = test_images.astype('float32') / 255
    test_images = np.asmatrix(test_images)
    test_images = test_images.transpose()
    test_labels = test_labels.transpose()
    train_data = (train_images[:,:49999], train_labels[:,:49999])
    valid_data = (train_images[:,50000:], train_labels[:,50000:])
    test_data = (test_images, test_labels)
    return (train_data, valid_data, test_data)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    (train_images, train_labels), (test_images, test_labels) = load_data()
    n = np.random.randint(len(train_images)) # choose an image randomly
    digit = train_images[n]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.title("Image #{}, label={}".format(n, train_labels[n]),\
        color='red', fontsize=20)
    plt.axis("off")
    plt.show()
