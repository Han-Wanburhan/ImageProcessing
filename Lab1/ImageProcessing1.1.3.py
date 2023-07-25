import cv2
import matplotlib.pyplot as plt
import numpy as np

Origi_image = cv2.imread('cat.jpg')
image = cv2.cvtColor(Origi_image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(Origi_image, cv2.COLOR_RGB2GRAY)

plt.subplot(1, 2, 1)
plt.imshow(image,cmap='gray')
print("Original",image)
plt.title('Original')

# def BitQuantize(img):
#     C,H,W = img.shape
#     for c in range(0,C):
#         for w in range(0,W):
#             for h in range(0,H):
#                 img[c][h][w] = (img[c][h][w]/255)*15
#     return img

def BitQuantize(img):
    H,W = img.shape
    for w in range(0,W):
        for h in range(0,H):
            img[h][w] = (img[h][w]/255)*15
    return img

plt.subplot(1, 2, 2)
plt.imshow(BitQuantize(image),cmap='gray')
print("Quantize",BitQuantize(image))
plt.title('Quantize')

plt.show()