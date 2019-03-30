import keras
import math
import os
import cv2
import numpy as np

from read_data import *
from model_vgg import Siamese_net
import shutil
SIZE=64
H=0
W=0
def get_data(path,map):
    img = cv2.imread(path)
    img = img.astype(np.float32)
    B, G, R = cv2.split(img)
    if map == 1:
        mean = np.load(
            "F:\\pc\\ChangeDetection\\Data\\DataAppend\\Original\\Original\\train\\arcgis_train_mean_new.npy")
    else:
        mean = np.load(
            "F:\\pc\\ChangeDetection\\Data\\DataAppend\\Original\\Original\\train\\bingmaps_train_mean_new.npy")
    B -= mean[0]
    G -= mean[1]
    R -= mean[2]
    img = cv2.merge([B, G, R])

    return img

def main(batch_size):
    count=0
    labels = []
    predicts = []
    flienames=[]
    f=open("F:/pc/ChangeDetection/Data/DataAppend/train/arcgis_test/list.txt")
    for line in f:
        root1='F:/pc/ChangeDetection/Data/DataAppend/train/arcgis_test'
        root2='F:/pc/ChangeDetection/Data/DataAppend/train/bingmaps_test'


        line=line.strip()
        labels.append(int(line.strip()[-5]))


        print(line)
        flienames.append(line)

        img1=get_data(root1+line,1)
        img2=get_data(root2+line,2)

        img1 = torch.from_numpy(np.expand_dims(np.transpose(img1, [2, 0, 1]), axis=0)).cuda()
        img2 = torch.from_numpy(np.expand_dims(np.transpose(img2, [2, 0, 1]), axis=0)).cuda()

        output, output_soft = model(img1, img2)
        predicts.append(output.data.max(1)[1].item())
        print(output.data.max(1)[1].item(),line[-5],output.data.max(1)[1].item()==int(line[-5]))
        if(output.data.max(1)[1].item()==int(line[-5])):
            count +=1
            newroots1='F:/pc/error_pytorch/multiscale_true_new1'
            newroots2='F:/pc/error_pytorch/bingmaps_multiscale_true'

            if  not os.path.exists(newroots1):
                os.makedirs(newroots1)
            if not os.path.exists(newroots2):
                os.makedirs(newroots2)
            shutil.copyfile(root1+line,newroots1+'/'+str(count)+'arcgis'+line[-5]+'.jpg')
            shutil.copyfile(root2+line,newroots1+'/'+str(count)+'bingmaps'+line[-5]+'.jpg')
        print(count)
