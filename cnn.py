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
# rotate popuations

# generator of 3x3 patches from the input image
def gen_patches():
    lastimg = None
    gen = lbm.main()
    for i in range(100):
        lastimg = next(gen)
    IX, IY = np.meshgrid(range(4,lbm.Nx-33), range(4,lbm.Ny-5))
    IX = lbm.X.flatten()
    IY = lbm.Y.flatten()
    #print(f"IX.shape={IX.shape}, IY.shape={IY.shape}")

    while lbm.stillrun:
        #print("gen_patches: getting next image")
        img = next(gen)
        #print(f"gen_patches: got next image; img.shape={img.shape}")
        totalsamps = len(IX)//4
        ix = np.random.permutation(len(IX))
        #print(f"len(ix)={len(ix)}; ix[:5]={ix[:5]}")
        for k in range(totalsamps):
            #print(f"ix[{k}]={ix[k]}")
            j = IX[ix[k]]
            i = IY[ix[k]]
            if i < 0 or i >= img.shape[0] - 2 or j < 0 or j >= img.shape[1] - 2:
                continue
            x = np.array([lastimg[i:i + 3, j:j + 3,:]])
            Y = np.array([img[i+1:i+2, j+1:j+ 2,:12]])
            #yield (x,Y); print(f"yielding {i},{j}: x.shape={x.shape}, Y.shape={Y.shape}")
            #print(f"yielding {i},{j}: x.shape={x.shape}, Y.shape={Y.shape}")
            #continue
            """
            # Lattice speeds / weights
            NL = 9
            idxs = np.arange(NL, dtype=np.int64)
            cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1], dtype=np.int64)
            cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1], dtype=np.int64)
            weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1
            #                     0  1  2  3  4  5  6  7  8
            rot90idxs = np.array([0, 3, 4, 5, 6, 7, 8, 1, 2], dtype=np.int64)
            rot90from = np.array([0, 7, 8, 1, 2, 3, 4, 5, 6], dtype=np.int64)

            '''
            # rot90idxs
            6   5   4       8   7   6
            7   0   3   =>  1   0   5
            8   1   2       2   3   4
            '''
            """
            #                     0  1  2  3  4  5  6  7  8
            fliplridx = np.array([0, 1, 8, 7, 6, 5, 4, 3, 2], dtype=np.int64)
            for r in range(4):
                yield (x.copy(),Y.copy()) #print(f"yield {i},{j}: x.shape={(x+0).shape}, Y.shape={Y.shape}")
                #print(f"x:\n{x[...,:2]}")
                # rotate the matrix around z axis
                x = np.rot90(x, axes=(1,2)).copy()
                # rotate the velocity vectors
                x[...,0],x[...,1] = -x[...,1],x[...,0]
                Y[...,0],Y[...,1] = -Y[...,1],Y[...,0]
                # move the populations to their new homes 90 degrees hence
                # we will use the "where each popuation came from" array
                # to figure out where each population goes to
                popns = x[...,3:12].copy()    
                popns[...,[1,2,3,4,5,6,7,8]] = popns[...,lbm.rot90from[1:]]
                x[...,4:12] = popns[...,1:9]

                popny = Y[...,3:12].copy()    
                popny[...,[1,2,3,4,5,6,7,8]] = popny[...,lbm.rot90from[1:]]
                Y[...,4:12] = popny[...,1:9]

            x = np.fliplr(x)
            x[...,0] = -x[...,0]
            Y[...,0] = -Y[...,0]
            popns = x[...,3:12].copy()
            popns[...,1:9] = popns[...,fliplridx[1:9]]
            x[...,4:12] = popns[...,1:9]

            popny = Y[...,3:12].copy()
            popny[...,1:9] = popny[...,fliplridx[1:9]]
            Y[...,4:12] = popny[...,1:9]


            for r in range(4):
                yield (x.copy(),Y.copy()) #print(f"yield {i},{j}: x.shape={(x+0).shape}, Y.shape={Y.shape}")
                x = np.rot90(x, axes=(1,2))
                x[...,0],x[...,1] = -x[...,1],x[...,0]
                popns = x[...,3:12].copy()
                popns[...,[1,2,3,4,5,6,7,8]] = popns[...,lbm.rot90from[1:]]
                x[...,4:12] = popns[...,1:9]

                popny = Y[...,3:12].copy()    
                popny[...,[1,2,3,4,5,6,7,8]] = popny[...,lbm.rot90from[1:]]
                Y[...,4:12] = popny[...,1:9]
        lastimg = img
    return

#datagen = tf.data.Dataset.from_generator(gen_patches, output_types=(tf.float32,tf.float32), output_shapes=((3, 3, 4),(1,1,3)))
#datagen = datagen.batch(1000)
#datagen = datagen.repeat()



# create dense model
def dense_model():
    inputs = tf.keras.Input(shape=(3, 3, 13))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(48, activation='relu')(x)
    x = tf.keras.layers.Dense(48, activation='relu')(x)
    x = tf.keras.layers.Dense(48, activation='relu')(x)
    x = tf.keras.layers.Dense(48, activation='relu')(x)
    x = tf.keras.layers.Dense(48, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    lastlayer = tf.keras.layers.Dense(12, activation='linear')(x)
    outputs = tf.keras.layers.Reshape((1, 1, 12))(lastlayer)
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

