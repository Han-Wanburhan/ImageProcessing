import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Read image file
image = cv2.imread('./Lab5/image/Lumyai.jpg')

# Define resize factor
Reduce_factors = [2,4,5,7, 8, 15]  # At least 3 values
Scale_factors = [1 / factor for factor in Reduce_factors]  # Calculate scale factors

# Define interpolation method
inter_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
interpolation_titles = ["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA"]

# Define fill methods
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

# Parameters for image augmentation
Npic = 10
rotation_range = 40
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
zoom_range = 0.2
horizontal_flip = True

# Prepare ImageDataGenerator with parameters for each fill_mode
video_frames = {fill_mode: [] for fill_mode in fill_methods}

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

    # Generate augmented images and add them to the respective fill_mode's frame list
    for i in range(Npic):
        batch = datagen.flow(np.array([image]), batch_size=1)
        augmented_image = np.clip(batch[0], 0, 255).astype('uint8')
        video_frames[fill_mode].append(augmented_image[0])

# Create a VideoWriter object to save the frames as a video with a slower frame rate
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
frame_rate = 1  # Adjust the frame rate as needed (frames per second)
out = cv2.VideoWriter('augmented_video_slow.avi', fourcc, frame_rate, (image.shape[1], image.shape[0]))

# Write frames from each fill_mode to the video
for fill_mode in fill_methods:
    for frame in video_frames[fill_mode]:
        out.write(frame)

# Release the video writer
out.release()

print("Slow-motion video saved successfully.")
