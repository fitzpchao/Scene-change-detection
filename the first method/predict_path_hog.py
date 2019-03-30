import keras
import math
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from read_data import *
from model_hog import Siamese_net
import pandas as pd
SIZE=256
H=0
W=0
def load_img(path,map):
    #print(path)
    img = cv2.imread(path)
    img = img.astype(np.float32)
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
    img = cv2.merge([B, G, R])
    return img

def main(batch_size):
    root1='F:/pc/changedetectiondata/XYIMG2011L13_L19/L19'
    root2='F:/pc/changedetectiondata/XYIMG2018L13_L19/L19'
    root3='F:/pc/changedetectiondata/hog_11'
    root4='F:/pc/changedetectiondata/hog_18'
    output_csv=[]
    table=pd.read_table("F:/pc/changedetectiondata/L19correct_train.csv",header=None,sep=',',index_col=None)
    imgnames=table.values
    model = Siamese_net()
    model.load_weights("F:/pc/change_detection_tf/checkpoints_FA_correct_hog/weights-improvement-12.hdf5")
    for i in range(imgnames.shape[0]):
        #img1=np.expand_dims(cv2.resize(imgs1[i],(64,64)),axis=0)
        #img2 = np.expand_dims(cv2.resize(imgs2[i],(64,64)), axis=0)
        print(imgnames[i][0])
        print(imgnames[i][1])
        names=imgnames[i][0].split('/')
        name=names[0]+names[1]
        if (os.path.exists(os.path.join(root1, imgnames[i][0][:-3] + 'jpg'))):
            img1 = load_img(os.path.join(root1, imgnames[i][0][:-3] + 'jpg'), 1)
        else:
            img1 = load_img(os.path.join(root1, imgnames[i][0][:-3] + 'png'), 1)

        if (os.path.exists(os.path.join(root2, imgnames[i][0][:-3] + 'jpg'))):
            img2 = load_img(os.path.join(root2, imgnames[i][0][:-3] + 'jpg'), 2)
        else:
            img2 = load_img(os.path.join(root2, imgnames[i][0][:-3] + 'png'), 2)
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
        img2 = img_to_array(img2)
        img2 = np.concatenate([img2, img2_2], axis=-1)
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        output = model.predict([img1,img2])
        label = np.array(np.squeeze(output))
        label[label<0.5]=0
        label[label>=0.5]=1
        print(np.round(np.squeeze(output)),np.squeeze(output),label)
        output_csv.append([imgnames[i][0],label])
    output_csv=np.array(output_csv)
    output_csv=pd.DataFrame(output_csv)
    output_csv.to_csv('predictFA_correct_hog_8.csv',header=None,index=None)




if __name__ == '__main__':
    main(16)