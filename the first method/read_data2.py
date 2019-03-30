# coding=utf-8
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
import random
import os
seed = 7
np.random.seed(seed)

img_w = 64
img_h = 64


def load_img(path,map):
    img = cv2.imread(path)
    img = np.array(img, dtype="float")
    B, G, R = cv2.split(img)
    if map == 1:
        mean = np.load(
            "F:/pc/ChangeDetection/Data/DataAppend/train/arcgis_mean.npy")
    else:
        mean = np.load(
            "F:/pc/ChangeDetection/Data/DataAppend/train/bingmaps_mean.npy")
    B -= mean[0]
    G -= mean[1]
    R -= mean[2]
    img_new = cv2.merge([B, G, R])
    return img_new


def get_train_val(filepath):
    train_set = []
    val_set = []
    set=[]
    #读取文件名
    f=open(filepath)
    count=0
    for line in f:
            set.append(line.strip())
    f.close()
    random.shuffle(set)
    train_set = set[:8000]
    val_set = set[8000:]
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
            url = data[i]
            batch += 1
            img1 = load_img(root1 + '/' + url,1)
            img1 = img_to_array(img1)
            train_data1.append(img1)
            #print(root2 + '/' + url)
            img2 = load_img(root2 + '/' + url,2)

            img2 = img_to_array(img2)
            train_data2.append(img2)
            label = int(url[-5])
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
        random.shuffle(data)
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img1 = load_img(root1 + '/' + url, 1)
            img1 = img_to_array(img1)
            train_data1.append(img1)
            img2 = load_img(root2 + '/' + url, 2)
            img2 = img_to_array(img2)
            train_data2.append(img2)
            label = int(url[-5])
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


