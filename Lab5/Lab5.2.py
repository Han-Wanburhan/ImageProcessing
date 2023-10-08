import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Read image file
image = cv2.imread('./Lab5/image/Lumyai.jpg')

# Define resize factor
Reduce_factors = [2, 8, 15]  # At least 3 values
Scale_factors = [1 / factor for factor in Reduce_factors]  # Calculate scale factors

# Define interpolation method
inter_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
interpolation_titles = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA"]

# Define fill method
fill_methods = ['constant', 'nearest', 'reflect', 'wrap']

# Function to add Gaussian noise to an image
def add_gaussian_noise(img):
    # Generate Gaussian noise
    mean = 0
    std = 25  # You can adjust the standard deviation as needed
    gaussian_noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    
    # Add the noise to the image
    noisy_img = np.clip(img.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)
    return noisy_img


# Create a subplot grid
num_rows = len(Scale_factors)
num_cols = len(inter_methods)
fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))

# Parameters for image augmentation
Npic = 10
rotation_range = 40
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True

# Prepare ImageDataGenerator with parameters
for fill_mode in fill_methods:
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        preprocessing_function=add_gaussian_noise,
        fill_mode=fill_mode
    )

    # Generate augmented images
augmented_images = []

for i in range(Npic):
    batch = datagen.flow(np.array([image]), batch_size=1)
    augmented_image = np.clip(batch[0], 0, 255).astype('uint8')
    augmented_images.append(augmented_image)

# Display the augmented images
plt.figure(figsize=(10, 5))
for i, augmented_image in enumerate(augmented_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(cv2.cvtColor(augmented_image[0], cv2.COLOR_BGR2RGB))
    plt.title(f'Fill Mode: {fill_mode}')
    plt.axis('off')

plt.tight_layout()
plt.show()