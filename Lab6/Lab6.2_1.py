import numpy as np
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

# Load the MobileNet model with pre-trained weights and exclude the top classification layer
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add new layers
x = base_model.output

# Global Average Pooling Layer
x = GlobalAveragePooling2D()(x)

# Add Dense layers
# Layer 1 with 1024 nodes and ReLU activation
x = Dense(1024, activation='relu')(x)

# Layer 2 with 1024 nodes and ReLU activation
x = Dense(1024, activation='relu')(x)

# Layer 3 with 512 nodes and ReLU activation
x = Dense(512, activation='relu')(x)

# Output layer with 3 nodes (3 classes) and Softmax activation
preds = Dense(3, activation='softmax')(x)

# Assign transfer base model + new layers to model
model = Model(inputs=base_model.input, outputs=preds)

# Freeze layers from the base MobileNet model up to index 86
for layer in model.layers[:86]:
    layer.trainable = False

# Unfreeze layers you added on top of the base model (layers from index 86 onwards)
for layer in model.layers[86:]:
    layer.trainable = True

# Display model summary
model.summary()

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing and image data generators
seed_value = 42  # Replace with your desired seed value
batch_size = 32  # Replace with your desired batch size

# Create DataGenerator objects
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
    fill_mode="nearest",
)

# Create Train Image generator
train_generator = datagen.flow_from_directory(
    './train/',  # Replace with your training data directory
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed_value,
    shuffle=True
)

# Create Validation Image generator
val_generator = datagen.flow_from_directory(
    './validate/',  # Replace with your validation data directory
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed_value,
    shuffle=True
)
