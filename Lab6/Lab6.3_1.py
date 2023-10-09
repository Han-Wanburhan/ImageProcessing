import numpy as np
import cv2
import matplotlib.pyplot as plt
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
seed_val = 123  # Replace with your validation seed value

# Create DataGenerator objects
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input,
    fill_mode="nearest",
)

# Create Train Image generator
train_generator = datagen.flow_from_directory(
    './Lab6/Train/',  # Replace with your training data directory
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed_value,
    shuffle=True
)

# Create Validation Image generator
val_generator = datagen.flow_from_directory(
    './Lab6/Validate/',  # Replace with your validation data directory
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed_val,
    shuffle=True
)

# Create Optimizer
opts = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opts, metrics=['accuracy'])
eps = 50
step_size_train = train_generator.n // train_generator.batch_size
step_size_val = val_generator.n // val_generator.batch_size

# check step_size_train = step_size_val -> if not, adjust batch_size to make them equal
if step_size_train != step_size_val:
    print("Warning: step_size_train is not equal to step_size_val.")
    new_batch_size = val_generator.n // step_size_train
    val_generator = datagen.flow_from_directory(
        './Lab6/Validate/',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=new_batch_size,
        class_mode='categorical',
        seed=seed_val,
        shuffle=True
    )
    step_size_val = val_generator.n // val_generator.batch_size
    print(f"Adjusted batch size to {new_batch_size} to make step_size_train equal to step_size_val.")

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=step_size_train,
                              validation_data=val_generator,
                              validation_steps=step_size_val,
                              epochs=eps,  # Use 'eps' instead of 'EP'
                              verbose=1)

# Performance Visualization
# Create a range of values for the x-axis (epochs)
epochs = range(1, eps + 1)  # Use 'eps' as the number of epochs

# View Accuracy (Training, Validation)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history["accuracy"], label="Train_acc")
plt.plot(epochs, history.history["val_accuracy"], label="Validate_acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

# View Loss (Training, Validation)
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label="Train_loss")
plt.plot(epochs, history.history['val_loss'], label="Validate_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Initial test generator
testPath = './Lab6/Test/'  # Replace with your test data directory
test_generator = datagen.flow_from_directory(
    testPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=1
)

# Get class id for y_real_class
y_true = test_generator.classes

# Predict images according to test_generator
preds = model.predict_generator(test_generator)
print(preds.shape)
print(preds)

# Get predicted class labels (argmax along axis 1)
y_pred = np.argmax(preds, axis=1)
print(y_true)
print(y_pred)

# Calculate confusion matrix and classification report
confusion = confusion_matrix(y_true, y_pred)
classification_rep = classification_report(y_true, y_pred)

print("Confusion Matrix:")
print(confusion)
print("\nClassification Report:")
print(classification_rep)

y_pred = np.argmax(preds,axis=1)
print(test_generator.classes)
print(y_pred)

# Calculate confusion matrix, classification report between y_true and df_class
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))