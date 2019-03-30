import cv2
import numpy as np

img1=cv2.imread("F:/pc/changedetectiondata/test_cut_2011.jpg")
img2=cv2.imread("F:/pc/changedetectiondata/test_cut_2018.jpg")
mask=cv2.imread('FA1.png',0)

h,w,_=img1.shape

for i in range(h):
    for j in range(w):
        if(mask[i,j]==255):
            img2[i,j,0] =img2[i,j,0]*0.7
            img2[i,j,1] =img2[i,j,1]*0.7 + 255 *0.3
            img2[i,j,2] =img2[i,j,2]*0.7 + 255 *0.3
            img1[i, j, 0] = img1[i, j, 0] * 0.7
            img1[i, j, 1] = img1[i, j, 1] * 0.7 + 255 * 0.3
            img1[i, j, 2] = img1[i, j, 2] * 0.7 + 255 * 0.3
cv2.imwrite('FA1_2011.png',img1)
cv2.imwrite('FA1_2018.png',img2)