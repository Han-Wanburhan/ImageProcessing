import cv2
import numpy as np
from matplotlib import pyplot as plt
img_1 = cv2.imread("./Lab2/img/Miso.jpg")
image_resized = cv2.resize(img_1, (200, 200))

output_file = "./Lab2/video/lab_2.1.2_V1_output_file.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = 1
frame = 1

vid_output = cv2.VideoWriter(output_file,fourcc, fps, (image_resized.shape[0],image_resized.shape[1]))
def GammaEquation(img):
    A = 1
    B = 0
    Y = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.11,1.12,1.13,1.14,1.15,1.16,1.17,1.18,1.19,1.20]
    H,W,C = img.shape
    result_img = np.zeros_like(img)
    for y in Y:
        for w in range(0,W):
            for h in range(0,H):
                for c in range(0,C):
                    result_img[h,w,c] = A*(img[h][w][c])**y+B
        vid_output.write(result_img)
    return result_img              
            

test = GammaEquation(image_resized)

plt.subplot(1, 2, 1)
plt.imshow(image_resized,cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(test,cmap='gray')
plt.title("Gramma")


        
print(image_resized,"\n","*"*50)
print(test)


plt.show()
vid_output.release()

