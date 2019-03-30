import pandas
import numpy as np
import re
import cv2
#img1=cv2.imread("C:/Users/fitzpc/Desktop/true_fasle_new/A0_change_pred.jpg")
#img2=cv2.imread("C:/Users/fitzpc/Desktop/true_fasle_new/A0_unchange_pred.jpg")
'''img1=cv2.imread("change_pred_mask.png",0)
img1[img1>0]=2
img2=cv2.imread("unchange_pred_mask.png",0)
img2[img2>0]=3
mask = cv2.imread('mask2.jpg',0)
mask[mask>0]=1
mask_new=img1+img2
mask_new =mask_new * mask
mask_new[mask_new>3]=0
img3=cv2.imread("change_true_mask.png",0)
img3[img3>0]=1
img4=cv2.imread("unchange_true_mask.png",0)
img4[img4>0]=4
mask_new2=img3+img4
mask_new2 =mask_new2 * mask
mask_new2[mask_new2>4]=0

reslut=mask_new2 + mask_new





cv2.imwrite('pred_mask.png',mask_new)
cv2.imwrite('true_mask.png',mask_new2)
cv2.imwrite('reslut.png',reslut)'''

img=cv2.imread('reslut.png',0)
img[img!=3]=0
img[img==3]=1
print(np.sum(img))
