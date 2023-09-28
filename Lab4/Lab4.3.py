#Array, image processing
import cv2
import numpy as np
# import matplotlib.pyplot as plt
#Model Operation
from keras import Model, Input
from keras.optimizers import Adam,SGD,RMSprop,Adadelta
import keras.utils as image
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from scipy import signal

# io
import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#ทำโครงข่ายของการประมาลผล หรือ การทำโมเดล Connect Encoder and Decoder Model
def create_autoencoder(optimizer= 'Adam',learning_rates = None):
    Input_img = Input(shape=(80,80,3))
    otp = None

    x1 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(Input_img)
    x2 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x1)
    x2 = MaxPool2D((2,2))(x2)
    x3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x2)
    encoded = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(x3)

    x4 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(encoded)
    x5 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x4)
    x5 = UpSampling2D((2,2))(x5)
    x6 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(x5)
    x7 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(x6)
    decoded = Conv2D(3, (3,3), padding = 'same')(x7)

    autoencoder = Model(Input_img, decoded)
    if optimizer == 'Adam':
        otp = Adam(learning_rates)
    elif optimizer == 'SGD':
        otp = SGD(learning_rates)
    elif optimizer == 'RMSprop':
        otp = RMSprop(learning_rates)
    elif optimizer == 'Adadelta':
        otp = Adadelta(learning_rates)

    autoencoder.compile(optimizer=otp, loss = 'mean_squared_error', metrics = ['mean_squared_error'])

    return autoencoder


intensity = [0, 1]
k = 42
scalar = 0.5
noise_mean = 0.3 
noise_std = scalar
noise_factor = scalar
ImgArray = []

#ส่วนของการอ่านภาพมาจากโฟเคอร์ที่เก็บภาพแล้วใส่ใน array ที่เราสร้างไว้
imgs = glob.glob('./Lab4/face_mini/**/*.jpg')
for img in imgs:
    loadImg = cv2.imread(img)
    ResizeImg = cv2.resize(loadImg, (80, 80))
    rgb_image = cv2.cvtColor(ResizeImg, cv2.COLOR_BGR2RGB)
    ImgArray.append(rgb_image)

#ส่วนของการทำ Nomirise
ImgsArray = np.array(ImgArray)
ImgsArray = ImgsArray / 255

train_x, test_x = train_test_split(ImgsArray, random_state=k, test_size=0.3)

train_x, val_x = train_test_split(train_x, random_state=k, test_size=0.2)


train_x_noise = train_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=train_x.shape))
val_x_noise = val_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=val_x.shape))
test_x_noise = test_x + (noise_factor * np.random.normal(loc=noise_mean, scale=noise_std, size=test_x.shape))


epoch = [150,200]
batch_size = [2,32]
optimizer = ['SGD', 'Adam']
learning_rates = [0.01, 0.001]


#การ test model ที่เราใช้งาน
#ใส่ model ที่ดราสร้างโดยที่ค่า batch_size จะอยู่ที่ 16 คือคือพารามิเตอร์ที่ใช้ในการกำหนดจำนวนของข้อมูลที่จะถูกนำเข้าไปในโมเดลเพื่อใช้ในกระบวนการฝึกสอน
#และใส่ค่า epoch หรือ จำนวนรอบของการสอน
model = KerasRegressor(build_fn=create_autoencoder, epochs=2, batch_size=16, verbose=0)


#ใช้ในการกำหนดค่าที่คุณต้องการใช้ในกระบวนการ Grid Search หรือการค้นหาค่า hyperparameter ที่ดีที่สุดสำหรับโมเดลกำหนดค่าการสอนในรูปแบบต่างๆ
param_grid = dict(batch_size=batch_size, epochs=epoch, optimizer=optimizer,learning_rates = learning_rates)

#ใช้สำหรับการทำค้นหา (hyperparameter tuning) ในโมเดลเครื่องจักร (machine learning) โดยใช้กระบวนการทดลองค่า hyperparameter ที่แตกต่างกันเพื่อหาค่า hyperparameter ที่ดีที่สุดสำหรับโมเดล
#n_jobs: คือจำนวนงาน (jobs) ที่คุณต้องการให้ GridSearchCV ทำงานพร้อมกัน การใช้ n_jobs ที่มากกว่า 1 จะช่วยเร่งกระบวนการ Grid Search แต่ควรระมัดระวังเรื่องทรัพยากรระบบ
#verbose: ระบุระดับของการแสดงผลในระหว่างกระบวนการค้นหา hyperparameter. ในที่นี้คุณใช้ verbose=10 เพื่อให้แสดงผลละเอียดมาก
#cv: คือจำนวนของ fold ใน cross-validation ที่คุณต้องการใช้ในกระบวนการค้นหา. ในที่นี้ใช้ cv=2 หมายถึงการใช้ K-Fold Cross Validation โดยแบ่งข้อมูลออกเป็น 2 ส่วน
#param_grid: คือ dictionary ที่ระบุค่าที่คุณต้องการทดลองสำหรับแต่ละ hyperparameter ในโมเดลของคุณ เช่นค่า batch_size, epochs, optimizer, และ learning_rate ที่ต้องการทดลอง
grid = GridSearchCV(estimator=model, n_jobs=1, verbose= 10, cv=4, param_grid=param_grid)
grid_result = grid.fit(train_x_noise, train_x)

best_params = grid_result.best_params_
best_score = grid_result.best_score_

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

#แสดงผลคะแนน
print(f'Best params : {best_params}')
print(f'Best score : {best_score}')

for mean,stdev,param in zip(means,stds,params):
    print("%f (%f) whit: %r" % (mean,stdev,param))



