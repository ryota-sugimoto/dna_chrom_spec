#!/usr/bin/env python

import os
import struct
import numpy as np
import scipy.io
from keras.utils import np_utils

data = scipy.io.loadmat("data.mat")

all_size = data["seq_tr"].shape[0]
train_size = 50000

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

out_n = 6
X_train = data["seq_tr"][:train_size].reshape(train_size,200,4)
Y_train = data["out_tr"][:train_size]
X_train_rc = reverse_complement(X_train)
X_train_r = reverse(X_train)
X_train_c = complement(X_train)
X_train = np.concatenate([X_train,X_train_rc])
Y_train = np.concatenate([Y_train]*2)
Y_train = Y_train[:,out_n]


X_test = data["seq_tr"][train_size:].reshape(all_size-train_size,200,4)
Y_test = data["out_tr"][train_size:]
Y_test = Y_test[:,out_n]

X_practice = data["seq_te"].reshape(10000,200,4)

import theano
theano.config.floatX = "float32"
X_train = X_train.astype(theano.config.floatX)
Y_train = Y_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)
Y_test = Y_test.astype(theano.config.floatX)

from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Dropout
from keras.layers.convolutional import Convolution1D,ZeroPadding1D
from keras.optimizers import SGD

np.random.seed(1)

model = Sequential()
model.add(ZeroPadding1D(12,input_shape=(200,4)))

model.add(Convolution1D(16, 3,
                        border_mode="same",
                        input_shape=(224,4),
                        subsample_length=2,
                        activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution1D(32, 3,
                        border_mode="same",
                        subsample_length=2,
                        activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution1D(64, 3,
                        border_mode="same",
                        subsample_length=2,
                        activation="relu"))
model.add(Dropout(0.25))
model.add(Convolution1D(128, 3,
                        border_mode="same",
                        subsample_length=2,
                        activation="relu"))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(224,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))


from keras.optimizers import sgd
model.compile(loss="mse",
              optimizer="adadelta",
              metrics=["accuracy"])

'''
for i,layer in enumerate(model.layers):
  print i,"input",layer.input
  print i,"output",layer.output
  print
exit(0)
'''

from numpy import vstack
from sklearn.metrics import roc_auc_score

for i in range(200):
  model.fit(X_train,
            Y_train,
            shuffle=True,
            nb_epoch=10,
            batch_size=200,
            verbose=1,
            validation_split=0.1)
  

  Y_train_pred = model.predict_proba(X_train, verbose=0)
  Y_test_pred = model.predict_proba(X_test, verbose=0)

  train_auc = roc_auc_score(Y_train, Y_train_pred)
  test_auc = roc_auc_score(Y_test, Y_test_pred)
  print "train_auc", train_auc
  print "test_auc", test_auc


Y_practice = model.predict_proba(X_practice, verbose=0)
res_f = open("result.txt","w")
for v in Y_practice:
  print >> res_f, " ".join(map(str,list(v)))
