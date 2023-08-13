import cv2
import numpy as np
import matplotlib.pyplot as plt

def GammaEquation(img, A , B):
    result_img = A * (img) + B
    result_img[result_img < 0] = 0
    result_img[result_img > 255] = 255
    return result_img.astype(np.uint8)

img_1 = cv2.imread("./Lab2/img/Miso.jpg")
image_resized = cv2.resize(img_1, (200, 200))

output_file = "./Lab2/video/lab_2.1.1_V2_output_file.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 1

vid_output = cv2.VideoWriter(output_file, fourcc, fps, (image_resized.shape[1], image_resized.shape[0]))

A = [0.5,1] 
B = [3,6,9,12,15,18,21,24,27,30] 

for a in A:
    for b in B:
        adjusted_img = GammaEquation(image_resized, a, b)
        vid_output.write(adjusted_img)

vid_output.release()

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2RGB))
plt.title("Gamma Corrected")

plt.show()