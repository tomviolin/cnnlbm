#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/18 15:58
# @Author  : tomh@uwm.edu

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import lbm
import csv
import keras.backend as K

# Functional model

def cnn_model():
    inputs = tf.keras.Input(shape=(lbm.Ny, lbm.Nx, 4))
    x0 = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(inputs)

    x1 = tf.keras.layers.Flatten()(x0)
    x2 = tf.keras.layers.Dense(128, activation='relu')(x1)
    x3 = tf.keras.layers.Dense(64, activation='relu')(x2)
    x4 = tf.keras.layers.Dense(32, activation='relu')(x3)
    x5 = tf.keras.layers.Dense(16, activation='relu')(x4)
    x6 = tf.keras.layers.Dense(8, activation='relu')(x5)
    x7 = tf.keras.layers.Dense(4, activation='relu')(x6)
    x8 = tf.keras.layers.Dense(2, activation='relu')(x7)
    x9 = tf.keras.layers.Dense(1, activation='relu')(x8)
    
    x10 = K.ones((lbm.Ny, lbm.Nx, 1))
    x11 = tf.keras.layers.Multiply()([x9, x10])
    outputs = K.add()([inputs, x11])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = cnn_model()
model.summary()
"""

    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(4, (3, 3), activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(4, (3, 3), activation='relu')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(4, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(4, (3, 3), activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Reshape((lbm.Ny, lbm.Nx, 4))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = cnn_model()
model.summary()



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', input_shape=(lbm.Ny, lbm.Nx, 4),padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(4, (3, 3), activation='relu'),
    tf.keras.layers.Conv2DTranspose(4, (3, 3), activation='relu'),
    tf.keras.layers.UpSampling2D(size=(2, 2)),
    tf.keras.layers.Conv2DTranspose(4, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.UpSampling2D(size=(2, 2)),
    tf.keras.layers.Conv2DTranspose(4, (3, 3), activation='relu', padding='same')])
model.summary()





    tf.keras.layers.Reshape((lbm.Ny, lbm.Ny, 4)),
    tf.keras.layers.Conv2DTranspose(4, (3, 3), activation='relu'),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')])

"""
