import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

Origi_image = cv2.imread('fish1.jpg')
image = cv2.cvtColor(Origi_image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(Origi_image, cv2.COLOR_RGB2GRAY)
image_resized = cv2.resize(image_gray, (200, 200))


x, y = np.meshgrid(np.arange(0, image_resized.shape[1]), np.arange(0, image_resized.shape[0]))
z = image_resized

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='gray')
print(x.shape,y.shape,z.shape)

plt.title('3D Surface Plot of Image')

plt.show()
