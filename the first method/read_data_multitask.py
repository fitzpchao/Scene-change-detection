# coding=utf-8
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
import random
import math
import pandas as pd
import os

img_w = 256
img_h = 256


def load_img(path,map):
    img = cv2.imread(path)
    img = np.array(img, dtype="float")
    '''B, G, R = cv2.split(img)
    B = (B - np.mean(B))
    G = (G - np.mean(G))
    R = (R - np.mean(R))
    img_new = cv2.merge([B, G, R])'''
    img = img / 255.0
    return img

def load_img_aug(path1,path2):
    img1=load_img(path1,1)
    img2=load_img(path2,2)
    rot_p = random.random()
    flip_p = random.random()
    if (rot_p < 0.5):
        pass
    elif (rot_p >= 0.5):
        for k in range(3):
            img1[:, :, k] = np.rot90(img1[:, :, k])
            img2[:, :, k] = np.rot90(img2[:, :, k])
    if (flip_p < 0.25):
        pass
    elif (flip_p < 0.5):
        for k in range(3):
            img1[:, :, k] = np.fliplr(img1[:, :, k])
            img2[:, :, k] = np.fliplr(img2[:, :, k])
    elif (flip_p < 0.75):
        for k in range(3):
            img1[:, :, k] = np.flipud(img1[:, :, k])
            img2[:, :, k] = np.flipud(img2[:, :, k])
    elif (flip_p < 1.0):
        for k in range(3):
            img1[:, :, k] = np.fliplr(np.flipud(img1[:, :, k]))
            img2[:, :, k] = np.fliplr(np.flipud(img2[:, :, k]))
    return  img1,img2

def get_train_val(filepath_train,filepath_val):

    #读取文件名
    table1 = pd.read_table(filepath_train, sep=',', header=None)
    train_set = table1.values
    table1 = pd.read_table(filepath_val, sep=',', header=None)
    val_set = table1.values

    return train_set, val_set

def generateTrainData(batch_size,root1,root2,data=[]):
    #print 'generateData...'
    while True:
        train_data1 = []
        train_data2 = []
        train_label1 = []
        train_label2 = []
        random.shuffle(data)
        batch = 0
        for i in (range(len(data))):
            url = data[i][0]
            url2 = data[i][0]
            batch += 1

            img1_type = 'png'
            img2_type = 'png'

            if (os.path.exists(os.path.join(root1, url[:-3] + 'jpg'))):
                img1_type = 'jpg'
            if (os.path.exists(os.path.join(root2, url2[:-3] + 'jpg'))):
                img2_type = 'jpg'
            img1,img2 =load_img_aug(os.path.join(root1, url[:-3] + img1_type),os.path.join(root2, url2[:-3] + img2_type))
            img1 = img_to_array(img1)
            img2 = img_to_array(img2)
            img_in1=np.concatenate([img1,img2],axis=-1)
            img_in2 = np.concatenate([img1, img2, img1 - img2], axis=-1)
            train_data1.append(img_in1)
            train_data2.append(img_in2)
            label1 = int(data[i][1])
            label2 = int(str(data[i][2])[0])


            #print(label)
            train_label1.append(label1)
            train_label2.append(label2)

            if batch % batch_size==0:
                #print 'get enough bacth!\n'
                train_data1 = np.array(train_data1)
                train_data2 = np.array(train_data2)
                train_label1 = np.array(train_label1)
                train_label2 = np.array(train_label2)

                yield ([train_data1,train_data2],[train_label1,train_label2])
                train_data1 = []
                train_data2 = []
                train_label1 = []
                train_label2 = []
                batch = 0

def generateValidData(batch_size,root1,root2,data=[]):
    #print 'generateValidData...'
    while True:
        valid_data1 = []
        valid_data2 = []
        valid_label1 = []
        valid_label2 = []

        batch = 0
        for i in (range(len(data))):
            url = data[i][0]
            url2 = data[i][0]
            batch += 1
            img1_type = 'png'
            img2_type = 'png'

            if (os.path.exists(os.path.join(root1, url[:-3] + 'jpg'))):
                img1_type = 'jpg'
            if (os.path.exists(os.path.join(root2, url2[:-3] + 'jpg'))):
                img2_type = 'jpg'
            img1, img2 = load_img_aug(os.path.join(root1, url[:-3] + img1_type),
                                      os.path.join(root2, url2[:-3] + img2_type))
            img_in1 = np.concatenate([img1, img2], axis=-1)
            img_in2 = np.concatenate([img1, img2, img1 - img2], axis=-1)
            valid_data1.append(img_in1)
            valid_data2.append(img_in2)
            label1 = int(data[i][1])
            label2 = int(str(data[i][2])[0])

            # print(label)
            valid_label1.append(label1)
            valid_label2.append(label2)
            if batch % batch_size==0:
                valid_data1 = np.array(valid_data1)
                valid_label1 = np.array(valid_label1)
                valid_data2 = np.array(valid_data2)
                valid_label2 = np.array(valid_label2)
                yield ([valid_data1,valid_data2],[valid_label1,valid_label2])
                valid_data1 = []
                valid_data2 = []
                valid_label1 = []
                valid_label2 = []

                batch = 0


