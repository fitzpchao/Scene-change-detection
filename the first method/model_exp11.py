from keras.applications.resnet50 import ResNet50
import keras.layers as layers
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D,concatenate, Flatten, Dropout, Activation, Input,Dense,GlobalAveragePooling2D

from keras import metrics, losses

import keras

def getModel():
    resnet_model_notop=ResNet50(include_top=False,
             weights='imagenet',
             input_tensor=None,
             input_shape=(256,256,3)
             )

    out = resnet_model_notop.get_layer(index=-1).output #
    model_resnet = Model(inputs=resnet_model_notop.input,output= out )
    img1 = Input((256, 256, 3))
    img2 = Input((256, 256, 3))

    feature1 = model_resnet(img1)
    feature2 = model_resnet(img2)
    feature1 = layers.GlobalAveragePooling2D(name='avg_pool1')(feature1)
    out1 =  layers.Dense(1, activation='sigmoid', name='medense_3')(feature1)
    feature2 = layers.GlobalAveragePooling2D(name='avg_pool2')(feature2)
    out2 = layers.Dense(1, activation='sigmoid', name='medense_4')(feature2)
    feature  = concatenate([feature1,feature2],axis=-1)
    out = layers.Dense(512, activation='relu', name='medense_1')(feature)
    out = layers.Dense(1, activation='sigmoid', name='medense_2')(out)
    model=Model(input=[img1,img2],output=[out,out1,out2])
    model.summary()
    adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss=[losses.binary_crossentropy,losses.binary_crossentropy,losses.binary_crossentropy],
                  loss_weights=[0.7,0.15,0.15],
                  metrics=[metrics.binary_accuracy])
    return model