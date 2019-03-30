import pandas
import numpy as np
import re
import cv2
img1=cv2.imread("F:/pc/changedetectiondata/groundtruth_new/A0_change.jpg")
img2=cv2.imread("F:/pc/changedetectiondata/groundtruth_new/A0_unchange.jpg")

h,w,_=img2.shape
print(h,w)
mask1=np.zeros([h,w],np.uint8)
mask2=np.zeros([h,w],np.uint8)
for i in range(h):
    for j in range(w):
        if((img1[i,j,0] !=  0 or img1[i,j,1] != 0 or img1[i,j,2] !=  0) and
                (img1[i,j,0] !=  1 or img1[i,j,1] != 1 or img1[i,j,2] !=  1)):
            mask1[i,j]=255
        if ((img2[i, j, 0] != 0 or img2[i, j, 1] != 0 or img2[i, j, 2] != 0) and
                (img2[i, j, 0] != 1 or img2[i, j, 1] != 1 or img2[i, j, 2] != 1)):
            mask2[i, j] = 255
cv2.imwrite('change_true_mask.png',mask1)
cv2.imwrite('unchange_true_mask.png',mask2)