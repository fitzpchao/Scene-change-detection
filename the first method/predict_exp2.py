import keras
import math
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from read_data_exp2 import *
from keras.utils import multi_gpu_model
from model_exp2 import Siamese_net
import pandas as pd
from sklearn.metrics import confusion_matrix
SIZE=256
H=0
W=0
def load_img(path,map):
    img = cv2.imread(path)
    img = np.array(img, dtype="float") / 255.0

    return img


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
    model.load_weights("F:/pc/change_detection_tf/check_points/exp2/weights-improvement-04.hdf5")
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
        img1_1, img1_2, img1_3, img1_4 = img_aug(img1)
        img2_1, img2_2, img2_3, img2_4 = img_aug(img2)
        maska,_ = model.predict(np.concatenate([img1_1, img2_1,img1_1 - img2_1],axis=-1))
        maskb,_= model.predict(np.concatenate([img1_2, img2_2,img1_2 - img2_2],axis=-1))
        maskc,_ = model.predict(np.concatenate([img1_3, img2_3,img1_3 - img2_3],axis=-1))
        maskd,_ = model.predict(np.concatenate([img1_4, img2_4,img1_4 - img2_4],axis=-1))
        mask1 = maska + maskb + maskc + maskd
        print(mask1.shape)
        mask2 = mask1[0] + mask1[1]
        predicts = mask2 / 8.0
        predicts[predicts >= 0.5] = 1
        predicts[predicts < 0.5] = 0
        preds.append(predicts)
        trues.append(int(test_set[i][1]))
    preds=np.array(preds)
    trues=np.array(trues)
    preds=np.squeeze(preds).astype(np.uint8)
    confusion_matrixs=confusion_matrix(trues,preds)
    print(confusion_matrixs)
    table= pd.DataFrame(confusion_matrixs)
    savepath='confusion_matrixs'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    table.to_csv(savepath + '/exp2.csv',header=None,index=False)







if __name__ == '__main__':
    main()