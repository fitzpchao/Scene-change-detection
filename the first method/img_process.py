import cv2
import math
import numpy as np
import time
import os
from osgeo import gdalnumeric

SIZE = 256
STRIDE = 200

def imageSpilt(imgName="C:/Users/fitzpc/Desktop/timg.jpg",savePath="C:/Users/fitzpc/Desktop/spilt_img"):
    img=gdalnumeric.LoadFile(imgName)
    img=np.transpose(img,[1,2,0])
    h, w, _ = img.shape
    step=0
    imgs_array = np.zeros([SIZE, SIZE, 3], dtype=np.uint8)
    #labels_array = np.zeros([SIZE, SIZE], dtype=np.uint8)
    for i in range(0, h, STRIDE):
        for j in range(0, w, STRIDE):
            if (i + SIZE <= h and j + SIZE <= w):
                imgs_array = img[i:i + SIZE, j:j + SIZE,:]
                # labels_array=label[i:i+SIZE,j:j+SIZE]
            elif (i + SIZE <= h and j + SIZE > w):
                imgs_array = img[i:i + SIZE, w - SIZE:, :]
            elif (i + SIZE > h and j + SIZE <= w):
                imgs_array = img[h - SIZE:, j:j + SIZE, :]
            elif (i + SIZE > h and j + SIZE > w):
                imgs_array = img[h - SIZE:, w - SIZE:, :]
            step = step + 1
            cv2.imwrite(savePath + '/' + str(step) + '.png', imgs_array)
        # cv2.imwrite(labels_path + '/' + str(step) + '.png', labels_array)
    print('ok')
def imageSpilt_gray(imgName="C:/Users/fitzpc/Desktop/timg.jpg",savePath="C:/Users/fitzpc/Desktop/spilt_img"):
    img=gdalnumeric.LoadFile(imgName)
    h, w = img.shape
    step=0
    imgs_array = np.zeros([SIZE, SIZE], dtype=np.uint8)
    #labels_array = np.zeros([SIZE, SIZE], dtype=np.uint8)
    for i in range(0, h, STRIDE):
        for j in range(0, w, STRIDE):
            if (i + SIZE <= h and j + SIZE <= w):
                imgs_array = img[i:i + SIZE, j:j + SIZE]
                # labels_array=label[i:i+SIZE,j:j+SIZE]
            elif (i + SIZE <= h and j + SIZE > w):
                imgs_array = img[i:i + SIZE, w - SIZE:]
            elif (i + SIZE > h and j + SIZE <= w):
                imgs_array = img[h - SIZE:, j:j + SIZE]
            elif (i + SIZE > h and j + SIZE > w):
                imgs_array = img[h - SIZE:, w - SIZE:]
            step = step + 1
            cv2.imwrite(savePath + '/' + str(step) + '.png', imgs_array)
        # cv2.imwrite(labels_path + '/' + str(step) + '.png', labels_array)
    print('ok')

def imgJigsaw(imgName="C:/Users/fitzpc/Desktop/timg.jpg",imgsPath="C:/Users/fitzpc/Desktop/spilt_img",saveName="C:/Users/fitzpc/Desktop/timg1.png"):
    img = cv2.imread(imgName)
    h, w, _ = img.shape
    n_h = h // STRIDE
    n_w = w // STRIDE
    print(n_h,n_w)
    n_hceil = math.ceil(h / STRIDE)
    n_wceil = math.ceil(w / STRIDE)
    step = n_wceil * n_hceil
    print(n_hceil,n_wceil)
    imgsNameList=os.listdir(imgsPath)
    print(imgsNameList)
    imgsList=[]
    for i in range(len(imgsNameList)):
        imgsList.append(cv2.imread(imgsPath + '/' + str(i +1) + '.png'))

    imgs_array = np.array(imgsList,dtype=np.uint8)
    print(imgs_array.shape)
    #imgs_array = np.zeros([n_hceil * n_wceil, SIZE, SIZE, 3], dtype=np.uint8)
    img_recover = np.zeros([h, w, 3], dtype=np.uint8)
    for i in range(n_hceil - 1, -1, -1):
        for j in range(n_wceil - 1, -1, -1):
            print(i,j)
            if (((i* STRIDE + SIZE)<= h)and ((j* STRIDE + SIZE)<= w)):
                step -= 1
                img_recover[i * STRIDE:i * STRIDE + SIZE, j * STRIDE:j * STRIDE + SIZE, :] = imgs_array[step]
            elif (((i* STRIDE + SIZE)<= h)and ((j* STRIDE + SIZE)> w)):
                step -= 1
                #print(img_recover[i * STRIDE : i* STRIDE + SIZE, w - SIZE:, :].shape)
                img_recover[i * STRIDE : i* STRIDE + SIZE, w - SIZE:, :] = imgs_array[step]
            elif (((i* STRIDE + SIZE)> h)and ((j* STRIDE + SIZE)<= w)):
                step -= 1
                img_recover[h - SIZE:, j * STRIDE:j * STRIDE + SIZE, :] = imgs_array[step]
            elif (((i* STRIDE + SIZE)> h)and ((j* STRIDE + SIZE)> w)):
                step -= 1
                img_recover[h - SIZE:, w - SIZE:, :] = imgs_array[step]
    cv2.imwrite(saveName,img_recover)

path='F:/pc/changedetectiondata/Building change detection dataset'
savepath='F:/pc/changedetectiondata/Building change detection dataset256'
imglist=os.listdir(path)
print(imglist)
for i in range(len(imglist)):
    name=imglist[i]
    if not os.path.exists(savepath + '/' + name[:-4]):
        os.makedirs(savepath + '/' + name[:-4])
    if(i<2):
        imageSpilt(path + '/' + name,savepath + '/' + name[:-4])
    else:
        imageSpilt_gray(path + '/' + name,savepath + '/' + name[:-4])