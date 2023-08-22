import cv2
import numpy as np
import matplotlib.pyplot as plt

def GammaEquation(img, gamma):
    A = 1
    B = 0
    result_img = A * (img ** gamma) + B
    result_img = Quantize(result_img)
    return result_img.astype(np.uint8)

def Quantize (img):
    img = (img-np.min(img))/(np.max(img)-np.min(img))
    img = img*(2**8-1)
    return img

img_1 = cv2.imread("./Lab2/img/Miso.jpg")
# image_resized = cv2.resize(img_1, (200, 200))

output_file = "./Lab2/video/lab_2.1.2_V2_output_file.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 1

vid_output = cv2.VideoWriter(output_file, fourcc, fps, (img_1.shape[1], img_1.shape[0]))

Y_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.99]

for gamma in Y_values:
    adjusted_img = GammaEquation(img_1, gamma)
    vid_output.write(adjusted_img)

vid_output.release()

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB))
plt.title("Gamma Corrected")

print(img_1,"\n","*"*50)
print(adjusted_img)

plt.show()