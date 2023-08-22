import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
from numpy import expand_dims
from matplotlib import pyplot as plt

# Load the VGG16 model (optional, you can load it if you need it)
# model = VGG16()
# model.summary()
# kernels, biases = model.layers[1].get_weights()
# listparam = model.layers[1].get_config()



# Load and preprocess the image
image = cv2.imread('./Lab3/image/B.jpeg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB color space



image = cv2.resize(image, (224, 224))  # Resize to VGG16 input size

rs_image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))

img_mean = np.array([123.68, 116.779, 103.93])  # Mean pixel values
img_mean1 = np.array([103.93, 116.779, 123.68])  # Mean pixel values

print (image)
# print (img_mean1)
# rs_image1 = image - img_mean1  # Subtract mean values
# # print (rs_image1)
print("Preprocessed Image Pixel Values:")
print (rs_image)

brg_image = rs_image - img_mean # Subtract mean values
print ("AFTER = ", rs_image)
rgb_image1 = rs_image - img_mean1  # Subtract mean values

plt.subplot(1, 3, 1)  # 1 row, 2 columns, subplot 1
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 3, 2)  # 1 row, 2 columns, subplot 2
plt.imshow(brg_image[0,:,:,:])
plt.title('Preprocessed Image')

plt.subplot(1, 3, 3)  # 1 row, 2 columns, subplot 2
plt.imshow(rgb_image1[0,:,:,:])
plt.title('Preprocessed Image')

plt.show()