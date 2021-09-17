"""Keras implementation of SSD."""

import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPool2D
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.layers import BatchNormalization

from SegLink.depthwise_conv import DepthwiseConv2D
# from keras.applications.mobilenet import DepthwiseConv2D

# from ssd_layers import Normalize

def ssd300_body(x):
    
    source_layers = []
    
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=2, padding='same', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv9_2', activation='relu')(x)
    source_layers.append(x)
    
    return source_layers


def mobilenet_body(x):

    source_layers = []

    # conv0
    x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv1_1')(x)
    x = BatchNormalization()(x) # yes this should be here
    x = Activation('relu')(x)

    # conv1dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv2_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv1
    x = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv2_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv2dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(2,2), use_bias=False, name='conv3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv2
    x = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv3_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv3dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv4_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv3
    x = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv4_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv4dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(2,2), use_bias=False, name='conv5_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv4
    x = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv5_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv5dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv6_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv5
    x = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv6_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv6dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(2,2), use_bias=False, name='con7_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv6
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv7_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Loop  ----

    # conv7dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='con8_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv7
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv8_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv8dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='con9_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv8
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv9_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv9dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='con10_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv9
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv10_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv10dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='con11_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv10
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv11_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv11dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='con12_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv11
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv12_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    source_layers.append(x)

    # End loop ---

    # conv12dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(2,2), use_bias=False, name='con13_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv12
    x = Conv2D(filters=1024, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv13_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # conv13dw
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='con14_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv13
    x = Conv2D(filters=1024, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv14_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    source_layers.append(x)

    # conv14_1
    x = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv15_1', activation='relu')(x)
    # conv14_2
    x = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv15_2', activation='relu')(x)
    source_layers.append(x)

    # conv15_1
    x = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv16_1', activation='relu')(x)
    # conv15_2
    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv16_2', activation='relu')(x)
    source_layers.append(x)

    # # conv16_1
    x = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv17_1', activation='relu')(x)
    # conv16_2
    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv17_2', activation='relu')(x)
    source_layers.append(x)

    # # conv17_1
    x = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv18_1', activation='relu')(x) #this is kernel_size=64
    # conv17_2
    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', use_bias=False, name='conv18_2', activation='relu')(x) # this is kernel_size=128 
    source_layers.append(x)
    
    return source_layers

def hybrid_body(x):
    
    source_layers = []
    
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)

    # conv_4_1
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv4_1_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv4_1_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv_4_2
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv4_2_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv4_2_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv_4_3
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv4_3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv4_3_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)

    # conv_5_1
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv5_1_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv5_1_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv_5_2
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv5_2_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv5_2_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv_5_3
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv5_3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv5_3_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)

    #fc6
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='fc6_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=1024, kernel_size=1, strides=1, padding='same', use_bias=False, name='fc6_2')(x) # HOW TO ADD A DILATION RATE.... WHAT IS A DILATION RATE???
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)

    #fc7
    x = DepthwiseConv2D(kernel_size=1, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='fc7_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=1024, kernel_size=1, strides=1, padding='same', use_bias=False, name='fc7_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    source_layers.append(x)

    # conv_6_1
    x = DepthwiseConv2D(kernel_size=1, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv6_1_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv6_1_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # conv_6_2
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(2,2), use_bias=False, name='conv6_2_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv6_2_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    source_layers.append(x)


    # # Block 4
    # x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x) #HERE
    # x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x) #HERE
    # x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x) #HERE
    # source_layers.append(x)
    # x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # # Block 5
    # x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x) #HERE
    # x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x) #HERE
    # x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x) #HERE
    # x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # # FC6
    # x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x) #HERE This guy is 4M params
    # # FC7
    # x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x) #HERE
    # source_layers.append(x)
    # # Block 6
    # x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x) #HERE
    # x = Conv2D(512, 3, strides=2, padding='same', name='conv6_2', activation='relu')(x) #HERE
    # source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x) 
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv9_2', activation='relu')(x)
    source_layers.append(x)
    # Block 10 
    x = Conv2D(128, 1, strides=1, padding='same', name='conv10_1', activation='relu')(x)
    x = Conv2D(256, 4, strides=2, padding='same', name='conv10_2', activation='relu')(x)
    source_layers.append(x)
    
    return source_layers

def hybrid_one(x):
    
    source_layers = []
    
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)

    # conv_4_1
    # x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv4_1_1')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv4_1_2')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    # x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)

    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=False, name='conv5_3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv5_3_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=2, padding='same', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv9_2', activation='relu')(x)
    source_layers.append(x)
    # Block 10 
    x = Conv2D(128, 1, strides=1, padding='same', name='conv10_1', activation='relu')(x)
    x = Conv2D(256, 4, strides=2, padding='same', name='conv10_2', activation='relu')(x)
    source_layers.append(x)
    
    return source_layers

def ssd512_body(x):
    
    source_layers = []
    
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(x)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=2, padding='same', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv9_2', activation='relu')(x)
    source_layers.append(x)
    # Block 10 
    x = Conv2D(128, 1, strides=1, padding='same', name='conv10_1', activation='relu')(x)
    x = Conv2D(256, 4, strides=2, padding='same', name='conv10_2', activation='relu')(x)
    source_layers.append(x)
    
    return source_layers


def ssd512_trunc_body(x):
    
    source_layers = []
    
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    y = Conv2D(512, 3, strides=1, padding='same', name='conv5_3', activation='relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(y)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=2, padding='same', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv9_2', activation='relu')(x)
    source_layers.append(x)
    # Block 10 
    x = Conv2D(128, 1, strides=1, padding='same', name='conv10_1', activation='relu')(x)
    x = Conv2D(256, 4, strides=2, padding='same', name='conv10_2', activation='relu')(x)
    source_layers.append(x)
    
    return source_layers, y

def ssd512v2_trunc_body(x):
    
    source_layers = []
    
    # Block 1
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_1', activation='relu')(x)
    x = Conv2D(64, 3, strides=1, padding='same', name='conv1_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_1', activation='relu')(x)
    x = Conv2D(128, 3, strides=1, padding='same', name='conv2_2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_2', activation='relu')(x)
    x = Conv2D(256, 3, strides=1, padding='same', name='conv3_3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_2', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv4_3', activation='relu')(x)
    source_layers.append(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(x)
    # Block 5
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same', name='conv5_2', activation='relu')(x)
    x = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, strides=(1,1), use_bias=True, name='conv5_3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', use_bias=True, name='conv5_3_2')(x)
    x = BatchNormalization()(x)
    y = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(y)
    # FC6
    x = Conv2D(1024, 3, strides=1, dilation_rate=(6, 6), padding='same', name='fc6', activation='relu')(x)
    # FC7
    x = Conv2D(1024, 1, strides=1, padding='same', name='fc7', activation='relu')(x)
    source_layers.append(x)
    # Block 6
    x = Conv2D(256, 1, strides=1, padding='same', name='conv6_1', activation='relu')(x)
    x = Conv2D(512, 3, strides=2, padding='same', name='conv6_2', activation='relu')(x)
    source_layers.append(x)
    # Block 7
    x = Conv2D(128, 1, strides=1, padding='same', name='conv7_1', activation='relu')(x)
    x = ZeroPadding2D()(x)
    x = Conv2D(256, 3, strides=2, padding='valid', name='conv7_2', activation='relu')(x)
    source_layers.append(x)
    # Block 8
    x = Conv2D(128, 1, strides=1, padding='same', name='conv8_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv8_2', activation='relu')(x)
    source_layers.append(x)
    # Block 9
    x = Conv2D(128, 1, strides=1, padding='same', name='conv9_1', activation='relu')(x)
    x = Conv2D(256, 3, strides=2, padding='same', name='conv9_2', activation='relu')(x)
    source_layers.append(x)
    # Block 10 
    x = Conv2D(128, 1, strides=1, padding='same', name='conv10_1', activation='relu')(x)
    x = Conv2D(256, 4, strides=2, padding='same', name='conv10_2', activation='relu')(x)
    source_layers.append(x)
    
    return source_layers, y


def multibox_head(source_layers, num_priors, num_classes, normalizations=None):

    postfix = '' if num_classes == 21 else '_%i'%num_classes

    mbox_conf = []
    mbox_loc = []
    for i in range(len(source_layers)):
        x = source_layers[i]
        name = x.name.split('/')[0]
        
        # normalize
        # if normalizations is not None and normalizations[i] > 0:
        #   name = name + '_norm'
        #   x = Normalize(normalizations[i], name=name)(x)
            
        # confidence
        name1 = name + '_mbox_conf' + postfix
        x1 = Conv2D(num_priors[i] * num_classes, 3, padding='same', name=name1)(x)
        x1 = Flatten(name=name1+'_flat')(x1)
        mbox_conf.append(x1)

        # location
        name2 = name + '_mbox_loc' + postfix
        x2 = Conv2D(num_priors[i] * 4, 3, padding='same', name=name2)(x)
        x2 = Flatten(name=name2+'_flat')(x2)
        mbox_loc.append(x2)

    mbox_loc = concatenate(mbox_loc, axis=1, name='mbox_loc')
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)

    mbox_conf = concatenate(mbox_conf, axis=1, name='mbox_conf')
    mbox_conf = Reshape((-1, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    predictions = concatenate([mbox_loc, mbox_conf], axis=2, name='predictions')
    
    return predictions


# def SSD300(input_shape=(300, 300, 3), num_classes=21):
#     """SSD300 architecture.

#     # Arguments
#         input_shape: Shape of the input image.
#         num_classes: Number of classes including background.
    
#     # Notes
#         In order to stay compatible with pre-trained models, the parameters 
#         were chosen as in the caffee implementation.
    
#     # References
#         https://arxiv.org/abs/1512.02325
#     """
#     K.clear_session()
    
#     x = input_tensor = Input(shape=input_shape)
#     source_layers = ssd300_body(x)
    
#     # Add multibox head for classification and regression
#     num_priors = [4, 6, 6, 6, 4, 4]
#     normalizations = [20, -1, -1, -1, -1, -1]
#     output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations)
#     model = Model(input_tensor, output_tensor)

#     # parameters for prior boxes
#     model.image_size = input_shape[:2]
#     model.source_layers = source_layers
#     model.source_layers_names = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2']
#     # stay compatible with caffe models
#     model.aspect_ratios = [[1,2], [1,2,3], [1,2,3], [1,2,3], [1,2], [1,2]]
#     model.minmax_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
#     model.steps = [8, 16, 32, 64, 100, 300]
    
#     return model


# def SSD512(input_shape=(512, 512, 3), num_classes=21):
#   """SSD512 architecture.

#   # Arguments
#       input_shape: Shape of the input image.
#       num_classes: Number of classes including background.
    
#   # Notes
#       In order to stay compatible with pre-trained models, the parameters 
#       were chosen as in the caffee implementation.
    
#   # # References
#   #   https://arxiv.org/abs/1512.02325
#   # """
#   # K.clear_session()
    
#   # x = input_tensor = Input(shape=input_shape)
#   # source_layers = ssd512_body(x)
    
#   # # Add multibox head for classification and regression
#   # # num_priors = [4, 6, 6, 6, 6, 4, 4]
#   # # normalizations = [20, -1, -1, -1, -1, -1, -1]
#   # # num_priors = [4, 6, 6, 6, 6, 4]
#   # # normalizations = [20, -1, -1, -1, -1, -1]
#   # # output_tensor = multibox_head(source_layers, num_priors, num_classes, normalizations)
#   # output_tensor = multibox_head(source_layers, num_priors, num_classes)
#   # model = Model(input_tensor, output_tensor)

#   # # parameters for prior boxes
#   # model.image_size = input_shape[:2]
#   # model.source_layers = source_layers
#   # model.source_layers_names = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2', 'conv8_2', 'conv9_2', 'conv10_2']
#   # # model.source_layers_names = ['conv12_2', 'conv14_2', 'conv15_2_2', 'conv16_2', 'conv17_2_2', 'conv18_2_2']
#   # # stay compatible with caffe models
#   # model.aspect_ratios = [[1,2], [1,2,3], [1,2,3], [1,2,3], [1,2,3], [1,2], [1,2]]
#   # model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460), (460, 537)]
#   # model.steps = [8, 16, 32, 64, 128, 256, 512]
#   # model.aspect_ratios = [[1,2], [1,2,3], [1,2,3], [1,2,3], [1,2,3], [1,2]]
#   # model.minmax_sizes = [(35, 76), (76, 153), (153, 230), (230, 307), (307, 384), (384, 460)]
#   # model.steps = [8, 16, 32, 64, 128, 256]
    
#   return model

