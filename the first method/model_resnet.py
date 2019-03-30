from keras.applications.resnet50 import ResNet50
import keras.layers as layers
from keras.models import Model
from keras import metrics, losses
import keras
def model():
    resnet_model_notop=ResNet50(include_top=False,
             weights='imagenet',
             input_tensor=None,
             input_shape=(256,256,3)
             )

    out = resnet_model_notop.get_layer(index=-1).output #
    out = layers.GlobalAveragePooling2D(name='avg_pool')(out)
    out = layers.Dense(1, activation='sigmoid')(out)
    model = Model(inputs=resnet_model_notop.input,output= out )
    model.summary()
    adam = keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss=[losses.binary_crossentropy],
                  metrics=[metrics.binary_accuracy])
    return model