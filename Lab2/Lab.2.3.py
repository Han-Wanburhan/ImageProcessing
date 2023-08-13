import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread                         # load the image
from skimage.exposure import cumulative_distribution  # calculate the cumulative pixel value
import numpy as np                                    # reshape the image
import matplotlib.pyplot as plt                       # plot the result

img_1 = cv2.imread("./Lab2/img/Temp1.jpg")
image1_resized = cv2.resize(img_1, (200, 200))
img_2 = cv2.imread("./Lab2/img/Temp2.jpg")
image2_resized = cv2.resize(img_2, (200, 200))

im1b, im1g, im1r = cv2.split(image1_resized)
im2b, im2g, im2r = cv2.split(image2_resized)

pixels = np.arange(256)

def getCDF(image):
    cdf, bins = cumulative_distribution(image)
    cdf = np.insert(cdf, 0, [0]*bins[0])
    cdf = np.append(cdf, [1]*(255-bins[-1]))
    return cdf

def histMatch(cdfInput, cdfTemplate, imageInput):
    pixelValues = np.arange(256)
    new_pixels = np.interp(cdfInput, cdfTemplate, pixels)
    imageMatch = (np.reshape(new_pixels[imageInput.ravel()], imageInput.shape)).astype(np.uint8)
    return imageMatch

plt.subplot(3, 3, 1)
plt.imshow(cv2.cvtColor(image1_resized, cv2.COLOR_BGR2RGB))
plt.title("Input")
ax1 = plt.subplot(3, 3, 2)
ax2 = plt.subplot(3, 3, 3)
for i, c in enumerate('bgr'):
    hist = cv2.calcHist([image1_resized], [i], None, [256], [0, 256])
    cdf1 = np.cumsum(hist) / sum(hist)  # Calculate CDF
    print (np.cumsum(hist),"/",sum(hist),"=",cdf1)
    ax1.plot(hist, c)  # Histogram
    ax2.plot(cdf1, c)  # Cumulative Distribution Function
    ax2.set_ylabel("CDF")
    
plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(image2_resized, cv2.COLOR_BGR2RGB))
plt.title("Template")
ax3 = plt.subplot(3, 3, 5)
ax4 = plt.subplot(3, 3, 6)
for i, c in enumerate('bgr'):
    hist = cv2.calcHist([image2_resized], [i], None, [256], [0, 256])
    cdf2 = np.cumsum(hist) / sum(hist)  # Calculate CDF
    print (np.cumsum(hist),"/",sum(hist),"=",cdf2)
    ax3.plot(hist, c)  # Histogram
    ax4.plot(cdf2, c)  # Cumulative Distribution Function
    ax4.set_ylabel("CDF")

image_result = np.zeros((image1_resized.shape)).astype(np.uint8)

for i in range(3):
    cdfInput = getCDF(image1_resized[:,:,i])
    cdfTemplate = getCDF(image2_resized[:,:,i])
    image_result[:,:,i] = histMatch(cdfInput, cdfTemplate, image1_resized[:,:,i])


plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
plt.title("Matching")
ax5 = plt.subplot(3, 3, 8)
ax6 = plt.subplot(3, 3, 9)
for i, c in enumerate('bgr'):
    hist = cv2.calcHist([image_result], [i], None, [256], [0, 256])
    cdf2 = np.cumsum(hist) / sum(hist)  # Calculate CDF
    print (np.cumsum(hist),"/",sum(hist),"=",cdf2)
    ax5.plot(hist, c)  # Histogram
    ax6.plot(cdf2, c)  # Cumulative Distribution Function
    ax6.set_ylabel("CDF")
    
cv2.imwrite('./Lab2/img/output/image_result.jpg', image_result)
plt.tight_layout()
plt.show()
