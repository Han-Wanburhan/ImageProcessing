import cv2
import matplotlib.pyplot as plt


image = cv2.imread('fish1.jpg')


bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(image)
plt.title('BGR')


plt.subplot(2, 4, 2)
plt.imshow(image[:,:,0],cmap='gray')
plt.title('B')

plt.subplot(2, 4, 3)
plt.imshow(image[:,:,1],cmap='gray')
plt.title('G')

plt.subplot(2, 4, 4)
plt.imshow(image[:,:,2],cmap='gray')
plt.title('R')


plt.subplot(2, 4, 5)
plt.imshow(bgr_image)
plt.title('RGP')


plt.subplot(2, 4, 6)
plt.imshow(bgr_image[:,:,0],cmap='gray')
plt.title('R')


plt.subplot(2, 4, 7)
plt.imshow(bgr_image[:,:,1],cmap='gray')
plt.title('G')


plt.subplot(2, 4, 8)
plt.imshow(bgr_image[:,:,2],cmap='gray')
plt.title('B')

plt.show()