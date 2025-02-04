import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw

from os import path

from utils import make_predictions, test_prediction

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


modelFilename = "model-i-2000-lr-0.15-20250204-213816.npz" # change acc. to model

base_path = path.dirname(__file__)
model_path = path.join(base_path, 'model', modelFilename)
# test_path = path.join(base_path, 'data' , 'test.csv')

# data = pd.read_csv(test_path)
# data = np.array(data).T / 255.

if not path.exists(model_path):
    print(f"Error: Model file not found! Make sure '{model_path}' exists.")
    exit()

loaded_weights = np.load(model_path)
W1 = loaded_weights["W1"]
b1 = loaded_weights["b1"]
W2 = loaded_weights["W2"]
b2 = loaded_weights["b2"]

# test_prediction(data, 3, W1, b1, W2, b2)

# Create a 28x28 drawing canvas
class DigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        # Canvas for drawing (280x280 to capture more details)
        self.canvas = tk.Canvas(root, width=280, height=280, bg="black")
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.draw)

        # Buttons
        self.btn_predict = tk.Button(root, text="Predict", command=self.predict_digit)
        self.btn_predict.pack()
        
        self.btn_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.btn_clear.pack()

        # Image buffer to store drawings
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)

    def draw(self, event):
        """Draws white lines on the black canvas."""
        x, y = event.x, event.y
        r = 10  # Radius of the pen
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def predict_digit(self):
        """Predicts the drawn digit using the model."""
        # Resize to 28x28
        img_resized = self.image.resize((28, 28))
        # plt.gray()
        # plt.imshow(img_resized, interpolation='nearest')
        # plt.show()
        img_array = np.array(img_resized) / 255.0  # Normalize
        img_array = img_array.reshape(28*28, -1)  # Reshape for model

        # Predict
        digit = make_predictions(img_array, W1, b1, W2, b2)

        # Show result
        result_label.config(text=f"Predicted: {digit}")

    def clear_canvas(self):
        """Clears the canvas."""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)

# Create GUI
root = tk.Tk()
digit_app = DigitRecognizer(root)

# Label for prediction output
result_label = tk.Label(root, text="Draw a digit and press 'Predict'", font=("Arial", 16))
result_label.pack()

root.mainloop()