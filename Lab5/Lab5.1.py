import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras import Model, Input
import keras.utils as image
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

from tensorflow.keras.datasets import fashion_mnist

from sklearn.model_selection import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Read image file
image = cv2.imread('./Lab5/image/Lumyai.jpg')

# Define resize factor
Reduce_factors = [2, 8, 15]  # At least 3 values
Scale_factors = [1 / factor for factor in Reduce_factors]  # Calculate scale factors

# Define interpolation method
inter_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
interpolation_titles = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA"]

# Create a subplot grid
num_rows = len(Scale_factors)
num_cols = len(inter_methods)
fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, scale_factor in enumerate(Scale_factors):
    for j, inter_method in enumerate(inter_methods):
        # Resize the image using the current scale factor and interpolation method
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=inter_method)
        
        # Format the scale factor and interpolation method with one decimal place
        scale_factor_str = f'{scale_factor:.1f}'
        
        # Display the resized image in the corresponding subplot
        axs[i, j].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        
        # Set the title with a smaller font size (e.g., fontsize=10)
        axs[i, j].set_title(
            f'Reduce_Factors{Reduce_factors[i]},Scale Factor: {scale_factor_str}\nInterpolation: {interpolation_titles[j]}',
            fontsize=10
        )
        axs[i, j].axis('off')

# Show the tabbed figure
plt.tight_layout()
plt.show()
