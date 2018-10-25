import json

from Nasnet import *
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
from keras.optimizers import SGD, Adam
import numpy as np
from keras import __version__ as keras_version
from distutils.version import StrictVersion
from keras.layers import Input
from keras.models import Model
from keras import regularizers
import math
import tensorflow as tf

MODEL_NAME = "NasNet"

K.set_image_dim_ordering('tf')

L2_WEIGHT_DECAY = 5e-5
DROPOUT=0.5

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# create the base pre-trained model
def get_model(class_size):
    input_tensor = Input(shape=(331,331,4),name='image_input')
    #input_tensor=input_tensor
    #new_model = NASNetMobile(weights='imagenet', include_top=False, input_tensor=input_tensor) 
    new_model = NASNetLarge(weights=None, include_top=False, input_tensor=input_tensor)
    new_model.summary()
    #x = Lambda(global_average_pooling, output_shape=global_average_pooling_shape)(new_model.layers[-2].output)
    #x = Dense(class_size, activation = 'sigmoid', kernel_initializer='uniform')(x)
    x = GlobalAveragePooling2D()(new_model.layers[-1].output)
    #x = Dense(class_size, activation = 'softmax', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.005))(x)  
    x = Dropout(DROPOUT)(x)
    x = Dense(class_size, kernel_regularizer=l2(L2_WEIGHT_DECAY), activation = 'sigmoid')(x)    
    
    # Create model.
    new_model = Model(input_tensor, x, name='nasnetCam')
    
    #for layer in new_model.layers[:-1]:
    #    layer.trainable = True
    
    for layer in new_model.layers:
        print('new layer name = ', layer.name, ' layer.trainable = ', layer.trainable)
        
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    #adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    new_model.compile(loss = 'binary_crossentropy', optimizer=sgd, metrics=[f1])
    return new_model

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer