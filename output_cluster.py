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

# from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import Dense, Dropout, Activation, Flatten, merge
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l1, l2, l1l2
from keras import backend as K
#from scipy.io import loadmat




def norm_l2(feats):
	feats = K.l2_normalize(feats, axis=-1)
	return feats

	
def norm_l2_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape) == 2
	return tuple(shape)



startN = 0
batch_size=20
nb_epoch = 1
nb_class = 500
# input feature dimensions
idt_dim = 396


inputs = Input(shape=(idt_dim, ))
ec = Dense(1000, activation='relu', trainable = False)(inputs)
ec = Dense(256, activation='relu')(ec)
ec = Dense(198, name='aux_output')(ec)
predictions = Dense(nb_class, W_regularizer=l2(1e-6))(ec)

model = Model(input=inputs, output=predictions)

weightName = './lit_svm_fl.h5'
model.load_weights(weightName)

model = Model(input=inputs, output=ec)
model.compile(loss='mse', optimizer='sgd')


for i in range(1, 11):
	fileName = './savedata/'+str(i).zfill(2)+'/hmdb_g1_iter1_4.mat'
	In = h5py.File(fileName,'r')
	X_train = np.array(In['data'][:])
	label = np.array(In['label'][:])
	del In
	X_train = X_train.astype('float32')
	X_train = np.transpose(X_train)
	print (X_train.shape)
	features = model.predict(X_train)
	savePath = './centers_'+str(i).zfill(2)+'.mat'
	sio.savemat(savePath, {'trainFeat': features, 'label': label})
