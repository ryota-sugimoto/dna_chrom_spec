#!/usr/bin/env python

import os
import struct
import numpy as np
import scipy.io
from keras.utils import np_utils

data = scipy.io.loadmat("data.mat")

all_size = data["seq_tr"].shape[0]
train_size = 50000

train_n = 3

comp_mat = np.matrix([[0,0,0,1],
                      [0,0,1,0],
                      [0,1,0,0],
                      [1,0,0,0]])

def reverse_complement(seqs):
  l = []
  for seq in seqs:
    a = np.array(seq*comp_mat)
    a = a[::-1]
    l.append(a)
  return np.array(l)

def reverse(seqs):
  l = []
  for seq in seqs:
    a = seq[::-1]
    l.append(a)
  return np.array(l)

def complement(seqs):
  l = []
  for seq in seqs:
    a = np.array(seq*comp_mat)
    l.append(a)
  return np.array(l)

X_train = data["seq_tr"][:train_size].reshape(train_size,200,4)
X_train_rc = reverse_complement(X_train)
X_train_r = reverse(X_train)
X_train_c = complement(X_train)
Y_train = data["out_tr"][:train_size]
Y_train = Y_train[:,train_n]

X_test = data["seq_tr"][train_size:].reshape(all_size-train_size,200,4)
X_test_rc = reverse_complement(X_test)
X_test_r = reverse(X_test)
X_test_c = complement(X_test)
Y_test = data["out_tr"][train_size:]
Y_test = Y_test[:,train_n]

import theano
theano.config.floatX = "float32"
X_train = X_train.astype(theano.config.floatX)
Y_train = Y_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)
Y_test = Y_test.astype(theano.config.floatX)

from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Dropout,Merge
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD

np.random.seed(1)

filter_n_1 = 16

model = Sequential()
model.add(Convolution1D(filter_n_1, 20,
                        border_mode="valid",
                        input_shape=(200,4),
                        subsample_length=2,
                        activation="relu"))
model.add(Dropout(0.25))

model_r = Sequential()
model_r.add(Convolution1D(filter_n_1, 20,
                          border_mode="valid",
                          input_shape=(200,4),
                          subsample_length=2,
                          activation="relu"))
model_r.add(Dropout(0.25))

model_c = Sequential()
model_c.add(Convolution1D(filter_n_1, 20,
                          border_mode="valid",
                          input_shape=(200,4),
                          subsample_length=2,
                          activation="relu"))
model_c.add(Dropout(0.25))

model_rc = Sequential()
model_rc.add(Convolution1D(filter_n_1, 20,
                          border_mode="valid",
                          input_shape=(200,4),
                          subsample_length=2,
                          activation="relu"))
model_rc.add(Dropout(0.25))

merged_model = Sequential()
merged_model.add(Merge([model,model_r,model_c,model_rc],
                        mode="concat",
                        concat_axis = 1))
merged_model.add(Convolution1D(64, 20,
                               border_mode="valid",
                               subsample_length=2,
                               activation="relu"))
merged_model.add(Dropout(0.25))
merged_model.add(Flatten())
merged_model.add(Dense(128, activation="relu"))
merged_model.add(Dropout(0.5))
merged_model.add(Dense(1, activation="sigmoid"))

merged_model.compile(loss="mse",
                     optimizer="adadelta",
                     metrics=["accuracy"])

merged_model.fit([X_train,X_train_r,X_train_c,X_train_rc],
                 Y_train,
                 nb_epoch=1,
                 batch_size=100,
                 verbose=1,
                 validation_split=0.1)


from numpy import vstack
from sklearn.metrics import roc_auc_score
Y_train_pred = merged_model.predict_proba([X_train,X_train_r,X_train_c,X_train_rc], verbose=0)
Y_test_pred = merged_model.predict_proba([X_test,X_test_r,X_test_c,X_test_rc], verbose=0)
print "train_auc", roc_auc_score(Y_train, Y_train_pred)
print "test_auc", roc_auc_score(Y_test, Y_test_pred)
