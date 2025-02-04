import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import os


from utils import gradient_descent, forward_prop, get_accuracy, get_prediction, download_data, save_model

base_path = os.path.dirname(__file__)
data_folder = "data"
train_path = os.path.join(base_path, data_folder , 'train.csv')
model_folder = "model"

Data_url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3004/861823/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1738938067&Signature=mC2qhHhy%2FqZQFxoQ0xewXMvin9O64rtJOxgq6QKrAQ8iVnz8hlz2PmVrQx7OpR%2B%2BtpLE6GDNcDqd3oqtRJgjlWcIrjiDBVRxfRBi6%2BmRTXvzvUj0JH5aKl7SNTndJiDmWUP2dTvgnISnSDX2EdWsI85DmGeXiUjNL9nQGsnxeN1fcx3Ma04lOLXWaFUkHdv3MxZGFOtMcZguA7A3Nmomyuo46ifkD07bbsFD6G6a8qqAxeRTb1XZqm5gLLTzyeKJJupoRP3tUjZHCgFKABkpF7%2FNHk%2B0Iv85mbbsEkyMnL9eb%2F81OZjk%2B3%2FJldcT0MboK2ZqlobHoItekZR1umn0eA%3D%3D&response-content-disposition=attachment%3B+filename%3Ddigit-recognizer.zip"

if os.path.exists(data_folder) and any(os.scandir(data_folder)):
    print("Data already exists. Skipping download.")
else:
    print("Data not found. Downloading...")

    download_data(base_path, data_folder, Data_url)

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

learning_rate = 0.15
iteration = 2000

# training Neural network
W1, b1, W2, b2 = gradient_descent(train_X, train_Y, iteration, learning_rate)

print("<--Testing data-->")
_, _, _, A2 = forward_prop(W1, b1, W2, b2, test_X)
print("Accuracy: ", get_accuracy(get_prediction(A2), test_Y))

weights = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2
}

metadata = {
    "learning_rate": learning_rate,
    "iteration": iteration
}

save_model(base_path, model_folder, weights, metadata=metadata)