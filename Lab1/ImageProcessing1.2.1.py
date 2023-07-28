import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('fish1.jpg')

# ===========================================RGB=================================================================
rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

plt.figure(figsize=(15, 10))

plt.subplot(4, 4, 1)
plt.imshow(rgb_image)
plt.title('RGB')


plt.subplot(4, 4, 2)
plt.imshow(rgb_image[:,:,0],cmap='gray')
plt.title('R')


plt.subplot(4, 4, 3)
plt.imshow(rgb_image[:,:,1],cmap='gray')
plt.title('G')


plt.subplot(4, 4, 4)
plt.imshow(rgb_image[:,:,2],cmap='gray')
plt.title('B')
# ==========================================End RGB==============================================================

# ===========================================HSV=================================================================
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

plt.subplot(4, 4, 5)
plt.imshow(hsv_image)
plt.title('RGB')


plt.subplot(4, 4, 6)
plt.imshow(hsv_image[:,:,0],cmap='gray')
plt.title('H')


plt.subplot(4, 4, 7)
plt.imshow(hsv_image[:,:,1],cmap='gray')
plt.title('S')


plt.subplot(4, 4, 8)
plt.imshow(hsv_image[:,:,2],cmap='gray')
plt.title('V')
# ==========================================End HSV==============================================================

# ===========================================HLS=================================================================
hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

plt.subplot(4, 4, 9)
plt.imshow(hls_image)
plt.title('HLS')


plt.subplot(4, 4, 10)
plt.imshow(hls_image[:,:,0],cmap='gray')
plt.title('H')


plt.subplot(4, 4, 11)
plt.imshow(hls_image[:,:,1],cmap='gray')
plt.title('L')


plt.subplot(4, 4, 12)
plt.imshow(hls_image[:,:,2],cmap='gray')
plt.title('S')
# ==========================================End HLS==============================================================

# ===========================================HLS=================================================================
YCrCB_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

plt.subplot(4, 4, 13)
plt.imshow(YCrCB_image)
plt.title('YCrCB')


plt.subplot(4, 4, 14)
plt.imshow(YCrCB_image[:,:,0],cmap='gray')
plt.title('Y')


plt.subplot(4, 4, 15)
plt.imshow(YCrCB_image[:,:,1],cmap='gray')
plt.title('Cr')


plt.subplot(4, 4, 16)
plt.imshow(YCrCB_image[:,:,2],cmap='gray')
plt.title('Cb')
# ==========================================End HLS==============================================================

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()