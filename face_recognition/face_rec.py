import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Input, Flatten, Conv2D, MaxPooling2D, Dense


def main():
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    
    # print(len(gpus))

    POS_PATH = os.path.join('data', 'positive')
    NEG_PATH = os.path.join('data', 'negative')
    ANC_PATH = os.path.join('data', 'archor')

    os.makedirs(POS_PATH)
    os.makedirs(NEG_PATH)
    os.makedirs(ANC_PATH)


    print(POS_PATH)
    print(NEG_PATH)
    print(ANC_PATH)


if __name__ == "__main__":
    main()