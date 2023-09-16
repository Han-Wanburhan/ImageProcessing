import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import Model, Input
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# # Function to load and preprocess images
# def load_and_preprocess_images(image_paths, target_size=(100, 100), noise_std=0.1, noise_factor=0.2):
#     images = []
#     for path in tqdm(image_paths):
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#         img = cv2.resize(img, target_size)  # Resize to target_size

#         # Normalize the image pixel values to [0, 1]
#         img = img / 255.0

#         # Add noise to the image
#         noise = np.random.normal(0, noise_std, img.shape).astype('float32')
#         noisy_img = img + noise_factor * noise

#         images.append((img, noisy_img))

#     return np.array(images)

# Function to load and preprocess images with increased noise
def load_and_preprocess_images(image_paths, target_size=(100, 100), noise_std=0.2, noise_factor=0.5):
    images = []
    for path in tqdm(image_paths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, target_size)  # Resize to target_size

        # Normalize the image pixel values to [0, 1]
        img = img / 255.0

        # Add more noise to the image
        noise = np.random.normal(0, noise_std, img.shape).astype('float32')
        noisy_img = img + noise_factor * noise

        # Clip pixel values to [0, 1] range
        noisy_img = np.clip(noisy_img, 0, 1)

        images.append((img, noisy_img))

    return np.array(images)

# Step 1: Load and preprocess images
image_paths = glob.glob("./Lab4/face_mini/*/*.jpg")  # Modify the path as needed
imgs_with_noise = load_and_preprocess_images(image_paths)

# Step 2: Split data into training and testing sets
train_data, test_data = train_test_split(imgs_with_noise, test_size=0.3, random_state=42)

# Step 3: Split training data into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Step 4: Define your deep learning model (autoencoder for denoising)
input_img = Input(shape=(100, 100, 3))

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train your model
autoencoder.fit(train_data[:, 1], train_data[:, 0],
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(val_data[:, 1], val_data[:, 0]),
                callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

# Step 6: Perform denoising on test data
denoised_test_data = autoencoder.predict(test_data[:, 1])

# Step 7: Display original, noisy, and denoised images
n = 10  # Number of images to display
plt.figure(figsize=(20, 6))

for i in range(n):
    # Display original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test_data[i, 0])
    plt.title("Original")
    plt.axis('off')

    # Display noisy images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(test_data[i, 1])
    plt.title("Noisy")
    plt.axis('off')

    # Display denoised images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_test_data[i])
    plt.title("Denoised")
    plt.axis('off')

plt.show()
