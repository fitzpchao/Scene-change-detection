# coding=utf-8
import argparse
import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from read_data_3d import *
from model_3d import C3D
seed = 7
np.random.seed(seed)


EPOCHS = 20
BS = 16
model = C3D(2,256,256,3)
fileroot = "checkpoints_3d"
filepath = os.path.join(fileroot, 'weights-improvement-{epoch:02d}.hdf5')
if (not os.path.exists(fileroot)):
    os.makedirs(fileroot)
modelcheck = ModelCheckpoint(filepath, monitor='val_acc', mode='max', verbose=1)
tb_cb = keras.callbacks.TensorBoard(log_dir='3d_log')
callable = [modelcheck, tb_cb]
train_set, val_set = get_train_val("F:/pc/changedetectiondata/L19_train_dataaug.csv",
                                   "F:/pc/changedetectiondata/L19_valid_dataaug.csv")
train_numb = len(train_set)
valid_numb = len(val_set)
print("the number of train data is", train_numb)
print("the number of val data is", valid_numb)
# model.load_weights('weights-improvement-10.hdf5')
model.fit_generator(generator=generateTrainData(BS,'F:/pc/changedetectiondata/XYIMG2011L13_L19/L19','F:/pc/changedetectiondata/XYIMG2018L13_L19/L19',train_set), steps_per_epoch=train_numb // BS,
                    validation_data=generateValidData(BS,'F:/pc/changedetectiondata/XYIMG2011L13_L19/L19','F:/pc/changedetectiondata/XYIMG2018L13_L19/L19', val_set), validation_steps=valid_numb // BS,
                    epochs=EPOCHS,
                    callbacks=callable, workers=1)
# plot the training loss and accuracy
# plot the training loss and accuracy