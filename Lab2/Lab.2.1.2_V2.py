import cv2
import numpy as np
import matplotlib.pyplot as plt

def GammaEquation(img, gamma):
    A = 1
    B = 0
    result_img = A * (img ** gamma) + B
    result_img[result_img < 0] = 0
    result_img[result_img > 255] = 255
    return result_img.astype(np.uint8)

img_1 = cv2.imread("./Lab2/img/Miso.jpg")
# image_resized = cv2.resize(img_1, (200, 200))

output_file = "./Lab2/video/lab_2.1.2_V2_output_file.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 1

vid_output = cv2.VideoWriter(output_file, fourcc, fps, (img_1.shape[1], img_1.shape[0]))

Y_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20]

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