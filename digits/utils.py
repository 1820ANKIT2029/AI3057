import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import requests
import zipfile
import os
from time import strftime

def download_data(base_path, data_folder, Data_url):
    zip_path = os.path.join(base_path, data_folder, "data.zip")

    # Ensure the folder exists
    os.makedirs(data_folder, exist_ok=True)

    # Download the ZIP file
    response = requests.get(Data_url, stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_folder)

        # Remove the ZIP file after extraction
        os.remove(zip_path)

        print(f"Data downloaded and extracted to '{data_folder}'")
    else:
        print("Failed to download the file")

def save_model(base_path, model_folder, weights, metadata=None):
    """Saves the trained NN weights and biases in a timestamped .npz file."""
    # Ensure the model directory exists
    model_dir = os.path.join(base_path, model_folder)
    os.makedirs(model_dir, exist_ok=True)

    filename = "model"
    if metadata["iteration"]:
        filename += ("-i-" + str(metadata["iteration"]))
    if metadata["learning_rate"]:
        filename += ("-lr-" + str(metadata["learning_rate"]))

    # Generate the model file path with timestamp
    model_path = os.path.join(model_dir, f'{filename}-{strftime("%Y%m%d-%H%M%S")}.npz')

    # Save weights to .npz file
    np.savez(model_path, **weights)

    print(f"Model saved at: {model_path}")
    return model_path

def init_params():
    """
    layer weights and bias

    W1, b1 -> layer 1  (10 neuron and each has 784(28x28) + 1 parameters)
    W2, b2 -> layer 2  (10 neuron and each has 10 + 1 parameters)
    """

    # random value b/w -0.5 and 0.5
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2

def Relu(Z):
    """
    Relu function: 
    zi is element i of Z
    Relu(zi) = 0 if zi <= 0 
            = zi otherwise
                
    """
    return np.maximum(0, Z)

def softmax(Z):
    """
    softmax function
    """
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    """
    Forward Propogation
    """
    Z1 = W1.dot(X) + b1
    A1 = Relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_Relu(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """
    Back propogation
    """
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_Relu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def get_prediction(A2):
    return np.argmax(A2, 0)

def get_accuracy(prediction, Y):
    print(prediction, Y)
    return np.sum(prediction == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_prediction(A2), Y))

    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_prediction(A2)
    return predictions

def test_prediction(data, index, W1, b1, W2, b2):
    current_image = data[:, index, None]
    prediction = make_predictions(data[:, index, None], W1, b1, W2, b2)
    # label = data_y[index]
    print("Prediction: ", prediction)
    # print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()