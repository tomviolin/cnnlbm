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
from keras.callbacks import TensorBoard

# rotate point (x,y) around (0,0) by 90 degrees
def rot90(x,y):
    return -y,x


# generator of 3x3 patches from the input image
def gen_patches():
    lastimg = None
    gen = lbm.main()
    while lbm.stillrun:
        totalsamps = 2000
        img = next(gen)
        for i in np.random.permutation(lbm.Ny - 2):
            for j in np.random.permutation(lbm.Nx - 2):
                if lastimg is not None:
                    x = np.array([lastimg[i:i + 3, j:j + 3, :]])
                    Y = np.array([img[i+1:i+2, j+1:j+ 2, :3]])
                    #yield (x,Y); print(f"yielding {i},{j}: x.shape={x.shape}, Y.shape={Y.shape}")
                    #print(f"yielding {i},{j}: x.shape={x.shape}, Y.shape={Y.shape}")
                    for r in range(4):
                        yield (x+0,Y); #print(f"yielding {i},{j}: x.shape={(x+0).shape}, Y.shape={Y.shape}")
                        x = np.rot90(x, axes=(1,2))
                        x[0,:,0,0],y[0,0,:,0]
                    x = np.fliplr(x)
                    for r in range(4):
                        yield (x+0,Y); #print(f"yielding {i},{j}: x.shape={(x+0).shape}, Y.shape={Y.shape}")
                        x = np.rot90(x, axes=(1,2))
                    totalsamps -= 1
                    if totalsamps <= 0:
                        break
            if totalsamps <= 0:
                break

        lastimg = img
    return
"""
datagen = tf.data.Dataset.from_generator(gen_patches, output_types=(tf.float32,tf.float32), output_shapes=((3, 3, 4),(3,3,4)))
datagen = datagen.batch(1000)
#datagen = datagen.repeat()
"""


# create dense model
def dense_model():
    inputs = tf.keras.Input(shape=(3, 3, 4))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(48, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.02)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(48, activation='relu')(x)
    x = tf.keras.layers.Dense(12, activation='relu')(x)
    lastlayer = tf.keras.layers.Dense(3, activation='linear')(x)
    outputs = tf.keras.layers.Reshape((1, 1, 3))(lastlayer)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = dense_model()
model.summary()

# compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])



# train model
model.fit(gen_patches(), epochs=10000, steps_per_epoch=1500, callbacks=[TensorBoard(log_dir='logs')])
now = datetime.datetime.now()

year = now.year
month = now.month
day = now.day
hour = now.hour
minute = now.minute
second = now.second
model.save(f'model_{year}_{month:02d}_{day:02d}_{hour:02d}_{minute:02d}_{second:02d}.h5')

