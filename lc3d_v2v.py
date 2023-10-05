'''Trains a simple local c3d.
Gets to xx% test accuracy after xx epochs.
xx seconds per epoch on a TITAN X GPU.
'''

from __future__ import print_function
import numpy as np
import scipy.io as sio
import random,cPickle
import sys   
import h5py
import os
import os.path
rootdir = "../hmdb51"
savepath = "../hmdb51_ours/01"

from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import Dense, Dropout, Activation, Flatten, merge
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l1, l2, l1l2
from keras import backend as K
#from scipy.io import loadmat




startN = 0
batch_size = 32
nb_epoch = 2

# input image dimensions
vid_rows, vid_cols = 32, 32 # already down sampled from 64 pixel
time_steps = 6 # already down sampled from 12 frames
# number of convolutional filters to use
nb_filters1 = 16
nb_filters2 = 64
nb_filters3 = 32
nb_filters4 = 64
nb_filters5 = 16

# size of pooling area for max pooling
# nb_pool = 2
# convolution kernel size
# nb_conv = 3
sub_num = 5000


# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

input_vid = Input(shape=(1, time_steps, vid_rows, vid_cols))

# encoder
conv1 = Convolution3D(nb_filters1, 3, 5, 5,
                        border_mode='same', activation='relu', dim_ordering='th')(input_vid)
pool1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), dim_ordering='th')(conv1)

conv2 = Convolution3D(nb_filters2, 3, 3, 3,
                        border_mode='same', activation='relu', dim_ordering='th')(pool1)
pool2 = MaxPooling3D((3, 2, 2), strides=(3, 2, 2), dim_ordering='th')(conv2)

conv3 = Convolution3D(nb_filters3, 1, 3, 3,
                        border_mode='same', dim_ordering='th')(pool2)
conv4 = Convolution3D(nb_filters3, 1, 3, 3,
                        border_mode='same', dim_ordering='th')(conv4)


# decoder

uppool2 = UpSampling3D(size=(3, 2, 2), dim_ordering='th')(conv4)
conv5 = Convolution3D(nb_filters6, 3, 3, 3,
                        border_mode='same', activation='relu', dim_ordering='th')(uppool2)

uppool1 = UpSampling3D(size=(2, 2, 2), dim_ordering='th')(conv4)



predict = Convolution3D(1, 3, 5, 5,
                        border_mode='same', dim_ordering='th')(uppool1)



model = Model(input=input_vid, output=predict)
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)


if startN:
	weightName = './weights/lc3d/lc3d_v2v_'+str(startN).zfill(3)+'.h5'
	model.load_weights(weightName)

model.compile(loss='conv_mse', optimizer=sgd)
for i in range(0,8): # split the input data in to 8 to save the memory
    dataname = './data_vid/'+'vol4ptHMDB_n_'+str(i+1)+'.mat'
    In = h5py.File(dataname,'r')
    X_train = np.array(In['allVid'][:])
    del In
    X_train = X_train.astype('float32')
    print (X_train.shape)
    np.random.shuffle(X_train)
    for q in range(0,4):
        print (q)
        for j in range(0, 4):
            if j == 14:
                model.fit(X_train[j*sub_num:,:,:,:,:], X_train[j*sub_num:,:,:,:,:], batch_size=batch_size, nb_epoch=nb_epoch,
                    shuffle=True, validation_split=0.001)
            else:
                model.fit(X_train[j*sub_num:(j+1)*sub_num+1,:,:,:,:], X_train[j*sub_num:(j+1)*sub_num+1,:,:,:,:], batch_size=batch_size, nb_epoch=nb_epoch, 
                    shuffle=True, validation_split=0.001)


weightName = './weights/lc3d/lc3d_v2v_'+str(startN+1).zfill(3)+'.h5'
model.save_weights(weightName)


