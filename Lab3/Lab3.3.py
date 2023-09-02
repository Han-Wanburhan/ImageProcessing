import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from matplotlib import pyplot as plt

# Load and preprocess the image
image = cv2.imread('./Lab3/image/A.jpg')
if image is None:
    raise ValueError("Failed to load the image.")

# Resize and convert to RGB
img = cv2.resize(image, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB color space
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Create a new Sequential model and add the 2D convolutional layer to it
new_model = Sequential()
new_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
new_model.summary()  # Print the updated model summary

# Plot the feature maps
feature_maps = new_model.predict(img)

for i in range(feature_maps.shape[3]):
    plt.subplot(8, 8, i + 1)  # Adjust the subplot layout as needed
    plt.imshow(feature_maps[0, :, :, i], cmap='gray')  # Display a single feature map
    plt.axis('off')
    plt.title("3.3")

plt.show()
