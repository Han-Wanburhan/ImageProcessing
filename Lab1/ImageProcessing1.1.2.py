import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

image = cv2.imread('fish1.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


## ========================================================= #
#                   Original
# ========================================================= #
print (rgb_image.shape)
plt.subplot(2, 4, 1)
plt.imshow(rgb_image[:,:,0])

ori_list = []
for i in image.shape:
    ori_list.append(i)
plt.title('BGR\n %s' %ori_list)

# ========================================================= #
#                   Transpose
# ========================================================= #
tp_image = np.transpose(rgb_image)
print (tp_image.shape)

plt.subplot(2, 4, 2)
plt.imshow(tp_image[0,:,:])

tp_list = []
for i in tp_image.shape:
    tp_list.append(i)
plt.title('Transpose\n %s' %tp_list)


# ========================================================= #
#                   Moveaxis
# ========================================================= #
mx_image = np.moveaxis(rgb_image,-1,0)
print (mx_image.shape)

plt.subplot(2, 4, 3)
plt.imshow(mx_image[0,:,:])

mx_list = []
for i in mx_image.shape:
    mx_list.append(i)
plt.title('Moveaxis\n %s' %mx_list)


# ========================================================= #
#                   Reshape
# ========================================================= #
rs_image = np.reshape(rgb_image,(rgb_image.shape[2],rgb_image.shape[0],rgb_image.shape[1]))
print(rs_image.shape)

plt.subplot(2, 4, 4)
plt.imshow(rs_image[0,:,:])

rs_list = []
for i in rs_image.shape:
    rs_list.append(i)
plt.title('Reshape\n %s' %rs_list)

# ========================================================= #

plt.show()