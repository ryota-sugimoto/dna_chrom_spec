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

out_n = 3
X_train = data["seq_tr"][:train_size].reshape(train_size,200,4)
Y_train = data["out_tr"][:train_size]
X_train_rc = reverse_complement(X_train)
X_train_r = reverse(X_train)
X_train_c = complement(X_train)
X_train = np.concatenate([X_train,X_train_rc,X_train_r,X_train_c])
Y_train = np.concatenate([Y_train]*4)
#Y_train = Y_train[:,out_n]


X_test = data["seq_tr"][train_size:].reshape(all_size-train_size,200,4)
Y_test = data["out_tr"][train_size:]
#Y_test = Y_test[:,out_n]

X_practice = data["seq_te"].reshape(10000,200,4)

import theano
theano.config.floatX = "float32"
X_train = X_train.astype(theano.config.floatX)
Y_train = Y_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)
Y_test = Y_test.astype(theano.config.floatX)

from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Dropout
from keras.layers.convolutional import Convolution1D,ZeroPadding1D,MaxPooling1D
from keras.constraints import maxnorm
from keras.optimizers import SGD

np.random.seed(1)

model = Sequential()
model.add(ZeroPadding1D(12,input_shape=(200,4)))
model.add(Dropout(0.1))

model.add(Convolution1D(64, 8,
                        border_mode="same",
                        W_constraint = maxnorm(2),
                        input_shape=(224,4),
                        subsample_length=4,
                        activation="relu"))
model.add(Dropout(0.25))

model.add(Convolution1D(128, 4,
                        border_mode="same",
                        W_constraint = maxnorm(2),
                        subsample_length=2,
                        activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(1024,
                activation="relu",
                W_constraint=maxnorm(2)))
model.add(Dropout(0.5))

model.add(Dense(512,
                activation="relu",
                W_constraint=maxnorm(2)))
model.add(Dropout(0.5))

model.add(Dense(8,  
                activation="sigmoid",
                W_constraint=maxnorm(2)))


from keras.optimizers import sgd
model.compile(loss="binary_crossentropy",
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
import sys

per_epoch = 1000

for i in range(3):
  model.fit(X_train,
            Y_train,
            shuffle=True,
            nb_epoch=per_epoch,
            batch_size=500,
            verbose=2,
            validation_split=0.1)
  

  Y_train_pred = model.predict_proba(X_train, verbose=0)
  Y_test_pred = model.predict_proba(X_test, verbose=0)
  train_auc,test_auc = [],[]
  for j in range(8):
    train_auc.append(roc_auc_score(Y_train[:,j], Y_train_pred[:,j]))
    test_auc.append(roc_auc_score(Y_test[:,j], Y_test_pred[:,j]))
  print "train_auc", " ".join(map(lambda f: "%0.03f"%(f,), train_auc))
  print "mean_train_auc", sum(train_auc)/float(len(train_auc))
  print "test_auc", " ".join(map(lambda f: "%0.03f"%(f,), test_auc))
  print "mean_test_auc",  sum(test_auc)/float(len(test_auc))
  sys.stdout.flush()

  model.save(str((i+1)*per_epoch)+"_epoch_"+"model.hdf")

Y_practice = model.predict_proba(X_practice, verbose=0)
res_f = open("result.txt","w")
for v in Y_practice:
  print >> res_f, " ".join(map(str,list(v)))
