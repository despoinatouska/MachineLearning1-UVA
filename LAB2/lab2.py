import time
import sys
import platform
from importlib.util import find_spec, module_from_spec
from sklearn.datasets import fetch_openml
import os
import pdb
import numpy as np
import matplotlib as plt


def plot_digits(data, num_cols, targets=None, shape=(28, 28)):
    num_digits = data.shape[0]
    num_rows = int(num_digits / num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(data[i].reshape(shape), interpolation='none', cmap='Greys')
        if targets is not None:
            plt.title(int(targets[i]))
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mnist = fetch_openml('mnist_784')
    data, target = mnist.data, mnist.target.astype('int')
    data = data.to_numpy()
    target = target.to_numpy()
    indices = np.arange(len(data))
    np.random.seed(123)
    np.random.shuffle(indices)
    data, target = data[indices].astype('float32'), target[indices]
    # Normalize the data between 0.0 and 1.0:
    data /= 255.
    # Split
    x_train, x_valid, x_test = data[:50000], data[50000:60000], data[60000: 70000]
    t_train, t_valid, t_test = target[:50000], target[50000:60000], target[60000: 70000]
    print(x_train.shape)
    plot_digits(x_train[0:40000:5000], num_cols=4, targets=t_train[0:40000:5000])

