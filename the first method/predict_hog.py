import keras
import math
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from read_data import *
from model_hog import Siamese_net
import pandas as pd
SIZE=64
H=0
W=0
def get_data(path,map):
    img = cv2.imread(path)
    img = img.astype(np.float32)
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
    img = cv2.merge([B, G, R])
    return img

def main(batch_size):
    model = Siamese_net()
    model.load_weights("F:/pc/change_detection_tf/checkpoints_hog/weights-improvement-01.hdf5")
    txt=pd.read_table("F:/pc/ChangeDetection/WH-ChangeDetection/ChangeDetection2080/1-Image/arcgis/list.txt",
                      header=None)
    filenames=txt.values
    print(filenames.shape)
    mask = []
    for i in range(filenames.shape[0]):
        img1_1=get_data('F:/pc/ChangeDetection/WH-ChangeDetection/ChangeDetection2080/1-Image/arcgis/'+filenames[i][0],1)
        img1_2=cv2.resize(cv2.imread('F:/pc/change_detection_tf/wh2080/arcgis_2080/'+filenames[i][0][:-4]+'.png'),(64,64))
        img1=np.concatenate([img1_1,img1_2],axis=-1)

        img2_1 = get_data(
            'F:/pc/ChangeDetection/WH-ChangeDetection/ChangeDetection2080/1-Image/bingmaps/' + filenames[i][0], 2)
        img2_2 = cv2.resize(cv2.imread('F:/pc/change_detection_tf/wh2080/bingmaps_2080/' + filenames[i][0][:-4] + '.png'),
                            (64, 64))
        img2 = np.concatenate([img2_1, img2_2], axis=-1)


        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)

        label=model.predict([img1,img2])
        label=np.squeeze(label,axis=0)
        print(label)
        if(label>0.5):
            label=1
        else:
            label=0
        mask.append([filenames[i][0],label])
    out_txt=pd.DataFrame(mask)
    out_txt.to_csv('label.csv',header=None,index=False,sep=' ')

if __name__ == '__main__':
    main(16)