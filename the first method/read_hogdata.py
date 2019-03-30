# coding=utf-8
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
import random
import math
import pandas as pd
import os
seed = 7
np.random.seed(seed)

img_w = 256
img_h = 256


def load_img(path,map):
    img = cv2.imread(path)
    img = np.array(img, dtype="float")
    B, G, R = cv2.split(img)
    if map == 1:
        mean = np.load(
            "F:/pc/changedetectiondata/2011BGR.npy")
    else:
        mean = np.load(
            "F:/pc/changedetectiondata/2018BGR.npy")
    B = (B - mean[0]) / math.sqrt(mean[3])
    G = (G - mean[1]) / math.sqrt(mean[4])
    R = (R - mean[0]) / math.sqrt(mean[5])
    img_new = cv2.merge([B, G, R])
    return img_new


def get_train_val(filepath_train,filepath_val):
    table1 = pd.read_table(filepath_train, sep=',', header=None)
    train_set = table1.values
    table1 = pd.read_table(filepath_val, sep=',', header=None)
    val_set = table1.values
    return train_set, val_set

def generateTrainData(batch_size,root1,root2,root3,root4,data=[]):
    #print 'generateData...'
    while True:
        train_data1 = []
        train_data2 = []
        train_label = []
        random.shuffle(data)
        batch = 0
        for i in (range(len(data))):
            url = data[i][0]
            batch += 1
            names=url.split('/')
            name=names[0] + names[1]
            if (os.path.exists(os.path.join(root1, url[:-3] + 'jpg'))):
                img1 = load_img(os.path.join(root1, url[:-3] + 'jpg'), 1)
            else:
                img1 = load_img(os.path.join(root1, url[:-3] + 'png'), 1)

            if (os.path.exists(os.path.join(root2, url[:-3] + 'jpg'))):
                img2 = load_img(os.path.join(root2, url[:-3] + 'jpg'), 2)
            else:
                img2 = load_img(os.path.join(root2, url[:-3] + 'png'), 2)

            if (os.path.exists(os.path.join(root3, name[:-3] + 'jpg'))):
                img1_2 = cv2.resize(cv2.imread(os.path.join(root3, name[:-3] + 'jpg')), (img_h, img_w))
            else:
                img1_2 = cv2.resize(cv2.imread(os.path.join(root3, name[:-3] + 'png')), (img_h, img_w))

            if (os.path.exists(os.path.join(root4, name[:-3] + 'jpg'))):
                img2_2 = cv2.resize(cv2.imread(os.path.join(root4, name[:-3] + 'jpg')), (img_h, img_w))
            else:
                img2_2 = cv2.resize(cv2.imread(os.path.join(root4, name[:-3] + 'png')), (img_h, img_w))
            img1 = img_to_array(img1)
            img1=np.concatenate([img1,img1_2],axis=-1)
            train_data1.append(img1_2)
            #print(root2 + '/' + url)
            img2 = img_to_array(img2)
            img2 = np.concatenate([img2, img2_2], axis=-1)
            train_data2.append(img2_2)
            label = int(data[i][1])
            #print(label)
            train_label.append(label)
            if batch % batch_size==0:
                #print 'get enough bacth!\n'
                train_data1 = np.array(train_data1)
                train_data2 = np.array(train_data2)
                train_label = np.array(train_label)
                yield ([train_data1,train_data2],train_label)
                train_data1 = []
                train_data2 = []
                train_label = []
                batch = 0

def generateValidData(batch_size,root1,root2,root3,root4,data=[]):
    #print 'generateValidData...'
    while True:
        train_data1 = []
        train_data2 = []
        train_label = []
        #random.shuffle(data)
        batch = 0
        for i in (range(len(data))):
            url = data[i][0]
            batch += 1
            names = url.split('/')
            name = names[0] + names[1]
            if (os.path.exists(os.path.join(root1, url[:-3] + 'jpg'))):
                img1 = load_img(os.path.join(root1, url[:-3] + 'jpg'), 1)
            else:
                img1 = load_img(os.path.join(root1, url[:-3] + 'png'), 1)

            if (os.path.exists(os.path.join(root2, url[:-3] + 'jpg'))):
                img2 = load_img(os.path.join(root2, url[:-3] + 'jpg'), 2)
            else:
                img2 = load_img(os.path.join(root2, url[:-3] + 'png'), 2)

            if (os.path.exists(os.path.join(root3, name[:-3] + 'jpg'))):
                img1_2 = cv2.resize(cv2.imread(os.path.join(root3, name[:-3] + 'jpg')), (img_h, img_w))
            else:
                img1_2 = cv2.resize(cv2.imread(os.path.join(root3, name[:-3] + 'png')), (img_h, img_w))

            if (os.path.exists(os.path.join(root4, name[:-3] + 'jpg'))):
                img2_2 = cv2.resize(cv2.imread(os.path.join(root4, name[:-3] + 'jpg')), (img_h, img_w))
            else:
                img2_2 = cv2.resize(cv2.imread(os.path.join(root4, name[:-3] + 'png')), (img_h, img_w))
            img1 = img_to_array(img1)
            img1 = np.concatenate([img1, img1_2], axis=-1)
            train_data1.append(img1_2)
            # print(root2 + '/' + url)
            img2 = img_to_array(img2)
            img2 = np.concatenate([img2, img2_2], axis=-1)
            train_data2.append(img2_2)
            label = int(data[i][1])
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data1 = np.array(train_data1)
                train_data2 = np.array(train_data2)
                train_label = np.array(train_label)
                yield ([train_data1, train_data2], train_label)
                train_data1 = []
                train_data2 = []
                train_label = []
                batch = 0


