import keras
import math
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from read_data_exp1 import *
from keras.utils import multi_gpu_model
from model_exp1 import Siamese_net
import pandas as pd
from sklearn.metrics import confusion_matrix
SIZE=256
H=0
W=0
def load_img(path,map):
    img = cv2.imread(path)
    img = np.array(img, dtype="float")
    print(img.shape)
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


def img_aug(img):
    img90 = np.array(np.rot90(img))
    img1 = np.concatenate([img[None], img90[None]])
    img2 = np.array(img1)[:, ::-1]
    img3 = np.array(img1)[:, :, ::-1]
    img4 = np.array(img2)[:, :, ::-1]
    return img1,img2,img3,img4
def main():
    model = Siamese_net()
    model = multi_gpu_model(model, 2)
    root1 = 'F:/pc/changedetectiondata/XYIMG2011L13_L19/L19'
    root2 = 'F:/pc/changedetectiondata/XYIMG2018L13_L19/L19'
    model.load_weights("F:/pc/change_detection_tf/check_points/exp1/weights-improvement-04.hdf5")
    preds=[]
    trues=[]
    table = pd.read_table("F:/pc/change_detection_tf/csv/cd_Building_test.csv", sep=',', header=None)
    test_set = table.values
    for i in range(test_set.shape[0]):
        url=test_set[i][0]
        img1_type = 'png'
        img2_type = 'png'
        if (os.path.exists(os.path.join(root1, url[:-3] + 'jpg'))):
            img1_type = 'jpg'
        if (os.path.exists(os.path.join(root2, url[:-3] + 'jpg'))):
            img2_type = 'jpg'
        path1=os.path.join(root1, url[:-3] + img1_type)
        path2=os.path.join(root2, url[:-3] + img2_type)

        img1 = load_img(path1, 1)
        img2 = load_img(path2, 2)
        predicts = model.predict([img1,img2])
        predicts[predicts >= 0.5] = 1
        predicts[predicts < 0.5] = 0
        preds.append(predicts)
        trues.append(int(str(test_set[i][2])[0]))
    preds=np.array(preds)
    trues=np.array(trues)
    preds=np.squeeze(preds).astype(np.uint8)
    confusion_matrixs=confusion_matrix(trues,preds)
    print(confusion_matrixs)
    table= pd.DataFrame(confusion_matrixs)
    savepath='confusion_matrixs'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    table.to_csv(savepath + '/exp1.csv',header=None,index=False)







if __name__ == '__main__':
    main()