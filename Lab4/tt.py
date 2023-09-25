import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import Model, Input
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# GPU configuration (optional)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Step 1: Read and preprocess image data
image_files = glob.glob("./Lab4/face_mini/**/*.jpg", recursive=True)
imgs = []

for fname in tqdm(image_files):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (100, 100))
    img = np.array(img)
    imgs.append(img)

imgs = np.array(imgs) / 255.0

# Step 2: Split data into training, validation, and test sets
random_state = 42
train_x, test_x = train_test_split(imgs, random_state=random_state, test_size=0.3)

# Step 3: Split training data into training and validation sets
train_x, val_x = train_test_split(train_x, random_state=random_state, test_size=0.2)

# Step 4: Define noise parameters
noise_mean = 0
noise_std = 0.5
noise_factor = 0.6

# Step 5: Add noise to the data
train_x_noise = train_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=train_x.shape))
val_x_noise = val_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=val_x.shape))
test_x_noise = test_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=test_x.shape))

# Step 6: Define the function to create the autoencoder model
def create_autoencoder(optimizer='adam', learning_rate=0.001, batch_size=16, epochs=20):
    # Define the encoder architecture
    Input_img = Input(shape=(100, 100, 3))
    x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)
    x1 = BatchNormalization()(x1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x3 = MaxPooling2D((2, 2), strides=(2, 2))(x2)
    encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x3)
    x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x5 = UpSampling2D((2, 2))(x4)
    x6 = Conv2D(128, (3, 3), activation='relu', padding='same')(x5)
    x6 = BatchNormalization()(x6)
    x7 = Conv2D(256, (3, 3), activation='relu', padding='same')(x6)
    x7 = BatchNormalization()(x7)

    # Define the decoder architecture
    decoded_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x7)

    # Create the autoencoder model
    autoencoder = Model(Input_img, decoded_img)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    return autoencoder

# Step 7: Define the parameter grid for hyperparameter tuning
opts = ['adam', 'sgd']
lnR = [0.001, 0.01, 0.1]
bs = [16, 32]
eps = [10, 30]

param_grid = dict(optimizer=opts, learning_rate=lnR, batch_size=bs, epochs=eps)

# Step 8: Create the KerasRegressor model
model = KerasRegressor(build_fn=create_autoencoder, verbose=0)

# Step 9: Create the GridSearchCV object
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2, verbose=10)

# Step 10: Fit the grid search to your data
grid_result = grid.fit(train_x_noise, train_x)

# Step 11: Print the best parameters and score
print("Best Parameters:", grid_result.best_params_)
print("Best Score:", grid_result.best_score_)

# Step 12: Get the mean and standard deviation of the scores for each parameter set
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, std, param in zip(means, stds, params):
    print(f"Mean: {mean}, Std: {std}, Params: {param}")

# Step 13: Train the autoencoder with the best hyperparameters
best_params = grid_result.best_params_
autoencoder = create_autoencoder(**best_params)

# Add EarlyStopping callback
callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = autoencoder.fit(train_x_noise, train_x, epochs=best_params['epochs'], batch_size=best_params['batch_size'],
                          shuffle=True, validation_data=(val_x_noise, val_x), verbose=1, callbacks=[callback])

# Step 14: Return the best parameters
best_params
