import cv2
import numpy as np
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from matplotlib import pyplot as plt

# Load the VGG16 model
model = VGG16()
model.summary()
kernels, biases = model.layers[1].get_weights()
listparam = model.layers[1].get_config()

# Load and preprocess the image
# image = cv2.imread('./Lab3/image/B.jpeg')
image = cv2.imread('./Lab3/image/A.jpg')
if image is None:
    raise ValueError("Failed to load the image.")

img = cv2.resize(image, (224, 224))  # Resize to VGG16 input size
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)

# Create a model that outputs the feature maps of the first convolutional layer
conv1_layer = model.layers[1]
model_conv1 = Model(inputs=model.inputs, outputs=conv1_layer.output)
model_conv1.summary()

# Extract feature maps
feature_maps = model_conv1.predict(img)

print('kernel = ', kernels[0][0], '\nbiases = ', biases)
print(listparam)

# # Display the original image
# plt.subplot(1, 2, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')

# # Display the preprocessed image
# plt.subplot(1, 2, 2)
# plt.imshow(img[0])
# plt.title('Preprocessed Image')

# plt.show()

# Plot the feature maps
plt.figure(figsize=(12, 12))
for i in range(feature_maps.shape[3]):
    plt.subplot(8, 8, i + 1)  # Adjust the subplot layout as needed
    plt.imshow(feature_maps[0, :, :, i],cmap='gray' )  # Display a single feature map
    plt.axis('off')
    plt.title(f'IMG {i + 1}')
plt.tight_layout()
plt.show()
