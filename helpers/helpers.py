from tensorflow.keras.datasets.mnist import load_data
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import randint

def get_size(dataset):
    print(f"Dataset Size: {len(dataset[0][0]) + len(dataset[1][0])}")
    print(f"Images have a shape: {dataset[0][0][0].shape}")

def get_dataset():
    dataset = load_data()
    get_size(dataset)

    return dataset

def create_train_test_validation_data(dataset):
    (X_train, y_train), (X_test, y_test) = dataset

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.20, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

    return X_train, y_train, X_test, y_test, X_val, y_val

def plot_random_image_sample(data, labels):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(data[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")

    plt.tight_layout()
    plt.show()

def plot_random_predicted_image_sample(X_test, predictions):
    for i in range(9):
        index = randint(1, 500)
        plt.subplot(330 + 1 + i)
        plt.imshow(X_test[index], cmap='gray')
        plt.title(f"Label: {np.argmax(predictions[index])}")

    plt.tight_layout()
    plt.show()