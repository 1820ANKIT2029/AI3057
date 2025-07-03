"""
lab7 

Ankit Kumar
A2
20233057
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import L2

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

"""# **Hyperparameter**"""

#config
batch_size = 16
learning_rate = 0.01
dropout_rate = 0.3
l2_reg = 0.01
epochs = 10

h1_size = 128
h2_size = 64

"""# **Data Loading**"""

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

"""# **Data Processing**"""

# Normalize pixel values to range [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten images from 28x28 to 784
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

"""# **Model Building**"""

# Build the neural network without L2 regularization and Dropout
model1 = keras.Sequential([
    Dense(h1_size, activation='relu'),
    Dense(h2_size, activation='relu'),
    Dense(10, activation='softmax')
])

# Build the neural network with L2 regularization and Dropout
model2 = keras.Sequential([
    Dense(h1_size, activation='relu', kernel_regularizer=L2(l2_reg)),
    Dropout(dropout_rate),
    Dense(h2_size, activation='relu', kernel_regularizer=L2(l2_reg)),
    Dropout(dropout_rate),
    Dense(10, activation='softmax')
])

"""# **Model training without dropout and L2 regularization**"""

# Compile the model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model1.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model1.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()

"""# **Model training with dropout and L2 regularization**"""

# Compile the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model2.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model2.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over epochs')
plt.show()