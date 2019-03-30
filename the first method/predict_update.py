import keras
import math
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from read_data import *
from model_vgg import Siamese_net
SIZE=64
STRIDE=32
H=0
W=0
def get_data(path,map):
    img = cv2.imread(path)
    #img = cv2.copyMakeBorder(img, 64, 64, 64, 64, cv2.BORDER_REFLECT)
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
    h, w, _ = img.shape
    n_h =(h - SIZE)// STRIDE +1
    n_w = (w - SIZE)//  STRIDE +1
    step = 0
    n_hceil = math.ceil((h - SIZE)/ STRIDE) + 1
    n_wceil = math.ceil((w - SIZE)/ STRIDE) + 1
    imgs_array = np.zeros([n_hceil * n_wceil, SIZE, SIZE, 3], dtype=np.float32)
    for i in range(n_hceil):
        for j in range(n_wceil):
            if (i != n_h and j != n_w):
                imgs_array[step] = img[i * STRIDE:i * STRIDE + SIZE, j * STRIDE:j * STRIDE +SIZE, :]
                step += 1
            elif (i != n_h and j == n_w):
                imgs_array[step] = img[i * STRIDE:i * STRIDE + SIZE, w - SIZE:, :]
                step += 1
            elif (i == n_h and j != n_w):
                imgs_array[step] = img[h - SIZE:, j * STRIDE:j * STRIDE +SIZE, :]
                step += 1
            elif (i == n_h and j == n_w):
                imgs_array[step] = img[h - SIZE:, w - SIZE:, :]
                step += 1
    return imgs_array,h,w,step,n_hceil,n_wceil,n_h,n_w

def main(batch_size):
    model = Siamese_net()
    model.load_weights("F:\pc\change_detection_tf\checkpoints\weights-improvement-07.hdf5")
    imgs1,h,w,step,n_hceil,n_wceil,n_h,n_w=get_data("F:/pc/ChangeDetection/WH-ChangeDetection/ChangeDetection2080/1-Image/arcgis_original.png",1)
    imgs2,_,_,_,_,_,_,_=get_data("F:/pc/ChangeDetection/WH-ChangeDetection/ChangeDetection2080/1-Image/bingmaps_original.png",2)
    mask = []
    for i in range(imgs1.shape[0]):
        img1=np.expand_dims(imgs1[i],axis=0)
        img2 = np.expand_dims(imgs2[i], axis=0)
        output = model.predict([img1,img2])
        mask.append(np.round(np.squeeze(output)))
        print(np.round(np.squeeze(output)),np.squeeze(output))
    h_orignal, w_orignal = h , w
    img_recover = np.zeros([h_orignal, w_orignal], dtype=np.uint8)
    for i in range(n_hceil - 1, -1, -1):
        for j in range(n_wceil - 1, -1, -1):
            if (i != n_h and j != n_w):
                step -= 1
                img_recover[i * STRIDE:(i + 1) * STRIDE, j * STRIDE:(j + 1) * STRIDE] = mask[step]
            elif (i != n_h and j == n_w):
                step -= 1
                img_recover[i * STRIDE:(i + 1) * STRIDE, w_orignal - STRIDE:] = mask[step]
            elif (i == n_h and j != n_w):
                step -= 1
                img_recover[h_orignal - STRIDE:, j * STRIDE:(j + 1) * STRIDE] = mask[step]
            elif (i == n_h and j == n_w):
                step -= 1
                img_recover[h_orignal - STRIDE:, w_orignal - STRIDE:] = mask[step]
    img_recover *= 255
    cv2.imwrite('changedetection_stride32.png',img_recover)



if __name__ == '__main__':
    main(16)