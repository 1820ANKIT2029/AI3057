import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from os import path

from utils import test_prediction

"""
loaded_weights = np.load(model_path)
W1 = loaded_weights["W1"]
b1 = loaded_weights["b1"]
W2 = loaded_weights["W2"]
b2 = loaded_weights["b2"]

test_prediction(data, 0, W1, b1, W2, b2)
test_prediction(data, 1, W1, b1, W2, b2)
test_prediction(data, 2, W1, b1, W2, b2)
test_prediction(data, 3, W1, b1, W2, b2)
"""


modelFilename = "model-20250202-155642.npz" # change acc. to model

base_path = path.dirname(__file__)
model_path = path.join(base_path, 'model', modelFilename)
test_path = path.join(base_path, 'data' , 'test.csv')

data = pd.read_csv(test_path)
data = np.array(data).T / 255.

loaded_weights = np.load(model_path)
W1 = loaded_weights["W1"]
b1 = loaded_weights["b1"]
W2 = loaded_weights["W2"]
b2 = loaded_weights["b2"]

test_prediction(data, 11, W1, b1, W2, b2)