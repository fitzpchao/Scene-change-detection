# coding=utf-8
import argparse
import numpy as np
import keras
from keras.engine.topology import Layer
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,concatenate, Flatten, Dropout, Activation, Input,Dense,GlobalAveragePooling2D
from keras.preprocessing.image import img_to_array
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import metrics,losses
from keras.layers.merge import concatenate
import keras.backend as K
from keras.utils import multi_gpu_model
import cv2
import random
import os

seed = 7
np.random.seed(seed)

img_w = 256
img_h = 256

n_label = 2

def divide(inputs=[]):
    output=inputs[0] /(inputs[1] + K.epsilon())

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


def attention(x):
    '''
    SE module performs inter-channel weighting.
    '''
    x_new=Conv2D(128,(1,1),activation='relu',padding='same')(x)
    x_pooling=MaxPooling2D((2, 2), strides=(2, 2))(x)
    conv1_1 = Conv2D(128, (7, 7), activation="relu", padding="same")(x_pooling)
    conv1_1_pooling=MaxPooling2D((2, 2), strides=(2, 2))(conv1_1)
    conv2_1 = Conv2D(128, (5, 5), activation="relu", padding="same")(conv1_1_pooling)
    conv2_1_pooling = MaxPooling2D((2, 2), strides=(2, 2))(conv2_1)
    conv3_1 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv2_1_pooling)
    conv1_2 = Conv2D(128, (7, 7), activation="relu", padding="same")(conv1_1)
    conv2_2 = Conv2D(128, (5, 5), activation="relu", padding="same")(conv2_1)
    conv3_2 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3_1)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv3_2), conv2_2], axis=3)
    print(up1.shape)
    conv4=Conv2D(64,(3,3),activation="relu", padding="same")(up1)
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1_2], axis=3)
    conv5=Conv2D(32,(3,3),activation="relu", padding="same")(up2)
    up3 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Conv2D(1,(3,3),activation="sigmoid", padding="same")(up3)
    scale = keras.layers.multiply([x_new, conv6])

    return scale

class MyLayer(Layer):
    def __init__(self,**kw):
        super(MyLayer,self).__init__(**kw)
    def build(self,input_shape):
        pass
    def call(self,x,mask=None):
        output=x[0]/(x[1]+K.epsilon())
        return output
    def get_output_shape_for(self,input_shape):
        pass



def Siamese_net():
    input1 = Input((img_h, img_w, 6))
    # Block 1
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(
        input1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x1)
    x1 = squeeze_excitation_layer(x1,64)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x1)

    # Block 2
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
        x1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
        x2)
    x2 = squeeze_excitation_layer(x2, 128)
    #x=attention(x)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x2)

    # Block 3
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
        x2)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
        x3)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
        x3)
    x3 = squeeze_excitation_layer(x3, 256)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x3)

    #Block 4
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
        x3)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
        x4)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
        x4)
    x4 = squeeze_excitation_layer(x4, 512)
    x4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x4)

    # Block 5
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
        x4)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
        x5)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
        x5)
    x5 = squeeze_excitation_layer(x5, 512)
    #x = attention(x)
    x5 = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x5)
    feature=Flatten()(x5)
    feature=Dropout(0.5)(feature)
    feature=Dense(1024,activation='relu')(feature)
    feature = Dropout(0.5)(feature)
    feature=Dense(1,activation='sigmoid')(feature)

    input2 = Input((img_h, img_w, 9))
    # Block 1
    x2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(
        input2)
    x2_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2_1)
    x2_1 = squeeze_excitation_layer(x2_1, 64)
    x2_1 = MaxPooling2D((2, 2), strides=(2, 2))(x2_1)
    x2_1 = concatenate([x2_1,x1])


    # Block 2
    x2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(
        x2_1)
    x2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(
        x2_2)
    x2_2 = squeeze_excitation_layer(x2_2, 128)
    # x=attention(x)
    x2_2 = MaxPooling2D((2, 2), strides=(2, 2))(x2_2)
    x2_2 = concatenate([x2_2,x2])

    # Block 3
    x2_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(
        x2_2)
    x2_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(
        x2_3)
    x2_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(
        x2_3)
    x2_3 = squeeze_excitation_layer(x2_3, 256)
    x2_3 = MaxPooling2D((2, 2), strides=(2, 2))(x2_3)
    x2_3 = concatenate([x2_3,x3])


    # Block 4
    x2_4 = Conv2D(512, (3, 3), activation='relu', padding='same')(
        x2_3)
    x2_4 = Conv2D(512, (3, 3), activation='relu', padding='same')(
        x2_4)
    x2_4 = Conv2D(512, (3, 3), activation='relu', padding='same')(
        x2_4)
    x2_4 = squeeze_excitation_layer(x2_4, 512)
    x2_4 = MaxPooling2D((2, 2), strides=(2, 2))(x2_4)
    x2_4 = concatenate([x2_4,x4])

    # Block 5
    x2_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(
        x2_4)
    x2_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(
        x2_5)
    x2_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(
        x2_5)
    x2_5 = squeeze_excitation_layer(x2_5, 512)
    # x = attention(x)
    x2_5 = MaxPooling2D((2, 2), strides=(2, 2))(x2_5)
    x2_5 = concatenate([x2_5,x5])
    feature2 = Flatten()(x2_5)
    feature2 = Dropout(0.5)(feature2)
    feature2 = Dense(1024, activation='relu')(feature2)
    feature2 = Dropout(0.5)(feature2)
    feature2 = Dense(512, activation='relu')(feature2)
    feature2 = Dropout(0.5)(feature2)
    feature2 = Dense(1, activation='sigmoid')(feature2)
    model = Model(inputs=[input1,input2], outputs=[feature2,feature])
    adam = keras.optimizers.Adam(lr=0.0001)
    model.summary()
    model = multi_gpu_model(model,gpus=2)
    model.compile(optimizer=adam, loss=[losses.binary_crossentropy,losses.binary_crossentropy],loss_weights=[0.5,0.5],metrics=[metrics.binary_accuracy])
    return model



