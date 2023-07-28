import cv2 as cv
import numpy as np
import cv2

img_1 = cv.imread("fish1.jpg")
img_2 = cv.imread("cat.jpg")
resize_img1 = cv.resize(img_1, (200, 200))
resize_img2 = cv.resize(img_2, (200, 200))

output_file = "lab_1.2.2_output_file.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

fps = 240
frame = 60 

vid_output = cv2.VideoWriter(output_file,fourcc, fps, (200,200))

w = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

for w1,w2 in zip(w,w[::-1]):
    result = cv2.addWeighted(resize_img1, w2, resize_img2, w1, 0)
    for i in range (frame):
        vid_output.write(result)

for w1,w2 in zip(w,w[::-1]):
    result = cv2.addWeighted(resize_img1, w1, resize_img2, w2, 0)
    for i in range (frame):
        vid_output.write(result)
        
vid_output.release()