import cv2
import numpy as np
import matplotlib.pyplot as plt

img_1 = cv2.imread("./Lab2/img/Miso.jpg")
image_resized = cv2.resize(img_1, (200, 200))

# Split the resized image into its color channels
b, g, r = cv2.split(image_resized)

# Apply histogram equalization to each color channel
equa_b = cv2.equalizeHist(b)
equa_g = cv2.equalizeHist(g)
equa_r = cv2.equalizeHist(r)

# Merge the equalized color channels back into an image
equa_colored = cv2.merge((equa_b, equa_g, equa_r))


plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
plt.title("Resized")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(equa_colored, cv2.COLOR_BGR2RGB))
plt.title("Equalized")

plt.subplot(2, 2, 2)
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    hist = cv2.calcHist([image_resized], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.title("Original Histogram")

plt.subplot(2, 2, 4)
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    hist = cv2.calcHist([equa_colored], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.title("Equlized Histogram")


plt.tight_layout()
plt.show()