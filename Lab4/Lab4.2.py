import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import Model, Input
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import warnings;
warnings.filterwarnings('ignore')

#GPU 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# 1. อ่านไฟล์ภาพทั้งหมดเก็บในรูป array (จำนวนภาพไม่น้อยกว่า 100 ภาพ)

image_files = glob.glob("./Lab4/face_mini/**/*.jpg",recursive=True)  # แทน path_to_images ด้วยที่ตั้งของไฟล์ภาพ
imgs = []

# 3. Append images to an array
for fname in tqdm(image_files):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # แปลงสีจาก BGR เป็น RGB
    img = cv2.resize(img, (100, 100))  # Resize ภาพเป็น (100, 100)
    img = np.array(img)
    imgs.append(img)

# 2. Normalized ภาพ (เพื่อให้ค่า pixel intensity = [0, 1])

imgs = np.array(imgs) / 255.0

# 4. แบ่งชุดข้อมูลเป็น Training_data, Testing_data (70 : 30)

random_state = 42  # กำหนด random state ตามที่คุณต้องการ
train_x, test_x = train_test_split(imgs, random_state=random_state, test_size=0.3)

# 5. แบ่งชุดข้อมูล Training_data เป็น Training_data, Validation_data (80:20)

train_x, val_x = train_test_split(train_x, random_state=random_state, test_size=0.2)

# 6. กำหนด noise parameters

noise_mean = 0
noise_std = 0.5  # ปรับค่าตามที่คุณต้องการ
noise_factor = 0.6  # ปรับค่าตามที่คุณต้องการ

# 7. สร้าง noise และเพิ่มเข้าในภาพ train_x, val_x, test_x

train_x_noise = train_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=train_x.shape))
val_x_noise = val_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=val_x.shape))
test_x_noise = test_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=test_x.shape))

# กำหนด Object แต่ละเลเยอร์ของ Encoder Architecture

Input_img = Input(shape=(100, 100, 3))

# Encoding architecture
x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(Input_img)

# Layer#2
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)

# Layer#3
x3 = MaxPooling2D((2, 2), strides=(2, 2))(x2)

# Layer#4
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x3)

# Layer#5
x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)

# Layer#6
x5 = UpSampling2D((2, 2))(x4)

# Layer#4
x6 = Conv2D(128, (3, 3), activation='relu', padding='same')(x5)

# Layer#8
x7 = Conv2D(256, (3, 3), activation='relu', padding='same')(x6)

# Layer#9
decoded_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x7)

# เลือกจะทำอะไรต่อในส่วนของ Decoder Architecture ตามต้องการ
autoencoder = Model(Input_img, decoded_img)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

callback = EarlyStopping(monitor= 'loss', patience=3)
history = autoencoder.fit(train_x_noise, train_x,
    epochs=20,
    batch_size=16,
    shuffle=True,
    validation_data=(val_x_noise, val_x),
    callbacks=[callback],
    verbose=1)

# Plot the training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


predictions_test = autoencoder.predict(test_x_noise)

n = 10
plt.figure(figsize=(50, 6))

for i in range(n):
    # Display original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test_x[i])
    plt.title("Original")
    plt.axis('off')

    # Display noisy images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(test_x_noise[i])
    plt.title("Noisy")
    plt.axis('off')

    # Display reconstructed images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(predictions_test[i])
    plt.title("Reconstructed")
    plt.axis('off')
    
plt.show()

