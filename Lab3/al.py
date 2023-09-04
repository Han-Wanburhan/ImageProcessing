import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from matplotlib import pyplot as plt
from scipy import signal

# Load the VGG16 model
model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Load and preprocess the image
image = cv2.imread('./Lab3/image/B.jpeg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))
img = img_to_array(image)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

# Get the weights (kernels) and biases from the first convolutional layer
layer = model.layers[1]
weights, biases = layer.get_weights()

# Perform convolution for each color channel and each kernel
img_result = np.zeros((1, 224, 224, len(biases)))

for i in range(len(biases)):  # Loop over each filter
    for channel in range(img.shape[-1]):  # Loop over each color channel
        img_result[:, :, :, i] += signal.convolve2d(
            img[0, :, :, channel], weights[:, :, channel, i], mode='same', boundary='fill', fillvalue=0
        )

# Add the biases to the convolutional results
for i in range(len(biases)):
    img_result[:, :, :, i] += biases[i]

# Apply ReLU activation function
img_result[img_result < 0] = 0

# Display the result of convolution and ReLU activation (64 feature maps)
plt.figure(figsize=(12, 12))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(img_result[0, :, :, i], cmap='gray')
    plt.axis('off')
    plt.title(f'Feature Map {i + 1}')

plt.tight_layout()
plt.show()
