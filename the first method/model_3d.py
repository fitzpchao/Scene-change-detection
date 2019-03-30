# coding=utf-8
import argparse
import numpy as np
import keras
from keras.layers import Conv2D, Conv3D,MaxPooling2D, MaxPooling3D,Reshape,UpSampling2D, \
    BatchNormalization, Reshape, Permute, Activation, Input,GlobalAveragePooling2D,Dense,\
    UpSampling3D,ZeroPadding3D,Flatten,Dense
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf
import cv2
import random
import os

def squeeze_excitation_layer(x, out_dim):
    '''
    SE module performs inter-channel weighting.
    '''
    squeeze = GlobalAveragePooling2D()(x)

    excitation = Dense(units=out_dim // 4)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = keras.layers.Reshape((1, 1, out_dim))(excitation)

    scale = keras.layers.multiply([x, excitation])

    return scale

def C3D(T,img_h,img_w,C):
    inputs = Input((T,img_h, img_w,C))

    conv1=Conv3D(32,kernel_size=(2,3,3),padding='same',activation="relu")(inputs)

    #conv1 = Conv3D(16, kernel_size=(3, 3, 3), padding='same',activation="relu")(conv1)
    pool1 = MaxPooling3D( pool_size=(1,2,2))(conv1)

    conv2 = Conv3D(64, kernel_size=(2, 3, 3), padding='same',activation="relu")(pool1)
    #conv2 = Conv3D(32, kernel_size=(3, 3, 3), padding='same',activation="relu")(conv2)
    pool2 = MaxPooling3D( pool_size=(1, 2, 2))(conv2)
    print(pool2.shape)

    conv3 = Conv3D(128, kernel_size=(2, 3, 3), padding='same',activation="relu")(pool2)
    conv3 = Conv3D(128, kernel_size=(2, 3, 3), padding='same',activation="relu")(conv3)
    pool3 = MaxPooling3D( pool_size=(1, 2, 2))(conv3)

    conv4 = Conv3D(256, kernel_size=(2, 3, 3), padding='same',activation="relu")(pool3)
    conv4 = Conv3D(256, kernel_size=(2, 3, 3), padding='same',activation="relu")(conv4)
    pool4 = MaxPooling3D( pool_size=(1, 2, 2))(conv4)

    conv5 = Conv3D(256, kernel_size=(2, 3, 3), padding='same', activation="relu")(pool4)
    print(conv5.shape)
    conv5 = Conv3D(256, kernel_size=(2, 3, 3), padding='same', activation="relu")(conv5)

    pool5 = MaxPooling3D(pool_size=(2, 2, 2))(conv5)

    feature = Flatten()(pool5)
    # feature=Dropout(0.5)(feature)
    feature = Dense(1024, activation='relu')(feature)
    # feature = Dropout(0.5)(feature)
    feature = Dense(512, activation='relu')(feature)
    # feature = Dropout(0.5)(feature)
    feature = Dense(1, activation='sigmoid')(feature)
    model = Model(inputs=inputs, outputs=feature)
    adam = keras.optimizers.Adam(lr=0.0001)
    model.summary()
    parallel_model = multi_gpu_model(model, gpus=2)



    model.compile(optimizer=adam, loss= 'binary_crossentropy', metrics=['binary_accuracy'])
    #model.compile(optimizer=adam, loss=[focal_loss(alpha=4)], metrics=['binary_accuracy'])
    #model.compile(optimizer=adam, loss=[combo_dice], metrics=['binary_accuracy'])
    return model