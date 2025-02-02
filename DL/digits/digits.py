import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from os import path, open, O_CREAT, close
from time import strftime

from utils import gradient_descent, forward_prop, get_accuracy, get_prediction

base_path = path.dirname(__file__)
train_path = path.join(base_path, 'data' , 'train.csv')
model_path = path.join(base_path, 'model', f'model-{strftime("%Y%m%d-%H%M%S")}.npz')
file = open(model_path, O_CREAT)
close(file)

# print(base_path)
# print(train_path)

data = pd.read_csv(train_path)

data = np.array(data)  # change pandas.DataFrames to numpy.array
m, n = data.shape

np.random.shuffle(data) # shuffle data to avoid biases and improve generalization

data_division = int(0.9*m)

# training data set
train_data = data[:data_division].T
train_X = train_data[1:n]
train_Y = train_data[0]
train_X = train_X / 255.   # normailization of data

# testing data set
test_data = data[data_division:].T
test_X = test_data[1:n]
test_Y = test_data[0]
test_X = test_X / 225.

print(train_Y.shape, train_X.shape)

# training Neural network
W1, b1, W2, b2 = gradient_descent(train_X, train_Y, 1000, 0.15)

print("<--Testing data-->")
_, _, _, A2 = forward_prop(W1, b1, W2, b2, test_X)
print("Accuracy: ", get_accuracy(get_prediction(A2), test_Y))

weights = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2
}

np.savez(model_path, **weights) # saving the trained NN weights and bias