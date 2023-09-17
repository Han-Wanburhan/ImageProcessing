# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import Model, Input
from keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define the path to the folder containing your image files
folder_path = './Lab4/face_mini/'

# Define the file pattern for the images you want to read (e.g., *.jpg)
file_pattern = '**/*.jpg'

# Use glob.glob to find all image files that match the pattern within the specified folder and its subdirectories
image_files = glob.glob(f'{folder_path}{file_pattern}', recursive=True)

# Initialize a list to store NumPy arrays of the images
images = []

# Loop through the image files, read them, and convert them to NumPy arrays
for file_path in image_files:
    # Read the image file using OpenCV
    img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

    # Resize the image to (80, 80) if needed
    img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_NEAREST)

    # Append the image to the list
    images.append(img)

# Convert the list to a NumPy array
images_array = np.array(images)

# Normalize the images
images_array = images_array / 255.0

# Split the data into training, validation, and testing sets
train_x, test_x = train_test_split(images_array, test_size=0.3, random_state=42)
train_x, val_x = train_test_split(train_x, test_size=0.2, random_state=42)

# Define noise parameters
noise_mean = 0
noise_std = 0.3
noise_factor = 0.6

# Create noise and add it to the images
train_x_noise = train_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=train_x.shape))
val_x_noise = val_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=val_x.shape))
test_x_noise = test_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=test_x.shape))

def create_model(optimizer='adam'):
    # Variable input Image
    input_img = Input(shape=(80, 80, 3))
    print(input_img.shape)

    # Encoding
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(input_img)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x2 = MaxPool2D((2, 2), strides=2)(x2)
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
    x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(x3)

    # Decoding
    x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(x4)
    x3 = Conv2D(128, (3, 3), activation='relu', padding='same')(x4)
    x2 = UpSampling2D((2, 2))(x3)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    decoded = Conv2D(3, (3, 3), padding='same')(x1)

    # Construct the autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    autoencoder.summary()
    return autoencoder

e = [2, 4, 6, 8, 16]
b = [16, 32, 64, 128]
autoencoder = create_model()

callback = EarlyStopping(monitor='loss', patience=3)
history = autoencoder.fit(train_x_noise, train_x,
                          epochs=10,
                          batch_size=16,
                          shuffle=True,
                          validation_data=(val_x_noise, val_x),  # Use val_x without noise
                          callbacks=[callback], verbose=1)

# Plot the training loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

num_of_plot = 5

for i in range(num_of_plot):
    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(test_x[i])

    # Noise Image
    plt.subplot(1, 3, 2)
    plt.title("Noise Image")
    plt.imshow(test_x_noise[i])

    # Denoise Image
    denoised_image = autoencoder.predict(np.expand_dims(test_x_noise[i], axis=0))
    plt.subplot(1, 3, 3)
    plt.title("Denoised Image")
    plt.imshow(denoised_image[0])

    plt.show()
