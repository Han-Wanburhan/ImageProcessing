import cv2
import numpy as np
from matplotlib import pyplot as plt
img_1 = cv2.imread("./Lab2/img/Miso.jpg")
image_resized = cv2.resize(img_1, (200, 200))

output_file = "./Lab2/video/lab_2.1.1_V1_output_file.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = 1
frame = 1

vid_output = cv2.VideoWriter(output_file,fourcc, fps, (image_resized.shape[0],image_resized.shape[1]))
# img_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
# img_gray1 = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
def LinearEquation(img):
    A = [0.5,1] 
    B = [3,6,9,12,15,18,21,24,27,30] 
    H,W,C = img.shape
    # H,W = img.shape
    result_img = np.zeros_like(img)
    for a in A:
        for b in B:
            for w in range(0,W):
                for h in range(0,H):
                    for c in range(0,C):
                        result_img[h,w,c] = a*(img[h][w][c])+b 
                #    result_img[h,w] = a*(img[h][w])+b
            vid_output.write(result_img)
    return result_img              
            
# test = LinearEquation(img_gray)

# plt.subplot(1, 2, 1)
# plt.imshow(cv2.resize(img_gray, (200, 200)),cmap='gray')
# plt.title("Original")

# plt.subplot(1, 2, 2)
# plt.imshow(test,cmap='gray')
# plt.title("Linear")

# print(img_gray,"\n","*"*50)
# print(test)

test = LinearEquation(image_resized)

plt.subplot(1, 2, 1)
plt.imshow(image_resized,cmap='gray')
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(test,cmap='gray')
plt.title("Linear")


        
print(image_resized,"\n","*"*50)
print(test)


plt.show()
vid_output.release()

