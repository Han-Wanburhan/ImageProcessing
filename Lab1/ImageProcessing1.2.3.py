import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('./Lab1/Image/cat.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
rgb_image = resize_img1 = cv2.resize(rgb_image, (200, 200))

plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.imshow(rgb_image)
plt.title('RGB')

rectangle = np.zeros((200, 200,3), dtype="uint8")
cv2.rectangle(rectangle, (80, 100), (160, 160), (255, 255, 255), -1)


plt.subplot(1, 3, 2)
plt.imshow(rectangle)
plt.title('Rectangel')

and_img = cv2.bitwise_and(rgb_image,rectangle)

plt.subplot(1, 3, 3)
plt.imshow(and_img)
plt.title('Bitwise_and')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()