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
    inputs = Input((img_h, img_w, 3))
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = squeeze_excitation_layer(x, 64)
    x1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    f1 = MaxPooling2D((16, 16), strides=(16, 16))(x1)
    f1 = Flatten()(f1)
    f1 = Dense(1024, activation='relu')(f1)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(
        x1)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(
        x)
    x = squeeze_excitation_layer(x, 128)
    # x=attention(x)
    x2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    f2 = MaxPooling2D((8, 8), strides=(8, 8))(x2)
    f2 = Flatten()(f2)
    f2 = Dense(1024, activation='relu')(f2)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(
        x2)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(
        x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(
        x)
    x = squeeze_excitation_layer(x, 256)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    x = attention(x)
    f3 = MaxPooling2D((4, 4), strides=(4, 4))(x3)
    f3 = Flatten()(f3)
    f3 = feature = Dense(1024, activation='relu')(f3)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(
        x3)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(
        x)
    x = squeeze_excitation_layer(x, 512)
    x4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(x)
    f4 = MaxPooling2D((2, 2), strides=(2, 2))(x4)
    f4 = Flatten()(f4)
    f4 = Dense(1024, activation='relu')(f4)
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(
        x4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(
        x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(
        x)
    x = squeeze_excitation_layer(x, 512)
    x5 = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    f5 = Flatten()(x5)
    f5 = Dense(1024, activation='relu')(f5)
    f6 = concatenate([f1, f2, f3, f4, f5])
    model_vgg = Model(inputs=inputs, outputs=f6)
    img1 = Input((img_h, img_w, 3))
    img2 = Input((img_h, img_w, 3))
    feature1 = model_vgg(img1)
    feature2 = model_vgg(img2)
    # feature_sub=keras.layers.Subtract()([feature1,feature2])
    # print(feature2.shape)
    # feature_div1=MyLayer()([feature2,feature1])
    # feature_div2 = MyLayer()([feature1, feature2])
    feature = concatenate([feature1, feature2])
    # feature=Flatten()(feature)
    # feature = Dropout(0.5)(feature)
    # feature = Dense(1024, activation='relu')(feature)
    # feature=Dropout(0.5)(feature)
    # feature=Dense(1024,activation='relu')(feature)
    feature = Dropout(0.5)(feature)
    feature = Dense(1024, activation='relu')(feature)
    feature = Dropout(0.5)(feature)
    feature = Dense(1, activation='sigmoid')(feature)
    model = Model(inputs=[img1, img2], outputs=feature)
    adam = keras.optimizers.Adam(lr=0.00001)
    model.summary()
    # model = multi_gpu_model(model,2)
    model.compile(optimizer=adam, loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    return model


