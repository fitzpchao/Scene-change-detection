import keras
import math
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from read_data import *
from model_vgg import Siamese_net
SIZE=256
H=0
W=0
def get_data(path,map):
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
    h, w, _ = img.shape
    n_h = h // SIZE
    y_h = h % SIZE
    n_w = w // SIZE
    y_w = w % SIZE
    step = 0
    n_hceil = math.ceil(h / SIZE)
    n_wceil = math.ceil(w / SIZE)
    imgs_array = np.zeros([n_hceil * n_wceil, SIZE, SIZE, 3], dtype=np.float32)
    for i in range(n_hceil):
        for j in range(n_wceil):
            if (i != n_h and j != n_w):
                imgs_array[step] = img[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE, :]
                step += 1
            elif (i != n_h and j == n_w):
                imgs_array[step] = img[i * SIZE:(i + 1) * SIZE, w - SIZE:, :]
                step += 1
            elif (i == n_h and j != n_w):
                imgs_array[step] = img[h - SIZE:, j * SIZE:(j + 1) * SIZE, :]
                step += 1
            elif (i == n_h and j == n_w):
                imgs_array[step] = img[h - SIZE:, w - SIZE:, :]
                step += 1
    return imgs_array,h,w,step,n_hceil,n_wceil,n_h,n_w

def main(batch_size):
    model = Siamese_net()
    model.load_weights("F:/pc/change_detection_tf/checkpoints_FA/weights-improvement-03.hdf5")
    imgs1,h,w,step,n_hceil,n_wceil,n_h,n_w=get_data("F:/pc/changedetectiondata/test_cut_2011.jpg",1)
    imgs2,_,_,_,_,_,_,_=get_data("F:/pc/changedetectiondata/test_cut_2018.jpg",2)
    mask = []
    for i in range(imgs1.shape[0]):
        #img1=np.expand_dims(cv2.resize(imgs1[i],(64,64)),axis=0)
        #img2 = np.expand_dims(cv2.resize(imgs2[i],(64,64)), axis=0)
        img1 = np.expand_dims(imgs1[i], axis=0)
        img2 = np.expand_dims(imgs2[i], axis=0)
        output = model.predict([img1,img2])
        label = np.array(np.squeeze(output))
        label[label<0.5]=0
        label[label>=0.5]=1
        mask.append(label)
        print(np.round(np.squeeze(output)),np.squeeze(output),label)
    img_recover = np.zeros([h, w], dtype=np.uint8)
    for i in range(n_hceil - 1, -1, -1):
        for j in range(n_wceil - 1, -1, -1):
            if (i != n_h and j != n_w):
                step -= 1
                img_recover[i * SIZE:(i + 1) * SIZE, j * SIZE:(j + 1) * SIZE] = mask[step]
            elif (i != n_h and j == n_w):
                step -= 1
                img_recover[i * SIZE:(i + 1) * SIZE, w - SIZE:] = mask[step]
            elif (i == n_h and j != n_w):
                step -= 1
                img_recover[h - SIZE:, j * SIZE:(j + 1) * SIZE] = mask[step]
            elif (i == n_h and j == n_w):
                step -= 1
                img_recover[h - SIZE:, w - SIZE:] = mask[step]
    img_recover *=255
    cv2.imwrite('FA3.png',img_recover)



if __name__ == '__main__':
    main(16)