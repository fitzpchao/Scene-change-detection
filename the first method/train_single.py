# coding=utf-8
import keras
from keras.callbacks import ModelCheckpoint
from read_data_nosiamese import *
from model_asoftmax import Siamese_net
path_checkpoints='F:/pc/change_detection_tf/check_points/asoftmax'

EPOCHS = 200
BS = 16
model = Siamese_net()

fileroot = path_checkpoints
filepath = os.path.join(fileroot, 'weights-improvement-{epoch:02d}.hdf5')
if (not os.path.exists(fileroot)):
    os.makedirs(fileroot)
modelcheck = ModelCheckpoint(filepath, monitor='val_acc', mode='max', verbose=1)
tb_cb = keras.callbacks.TensorBoard(log_dir='log/asoftmax')
callable = [modelcheck, tb_cb]
train_set, val_set = get_train_val("F:/pc/change_detection_tf/csv/cd_Building_train.csv",
                                   "F:/pc/change_detection_tf/csv/cd_Building_test.csv")
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