#!/usr/bin/env python 

import os
import struct
import numpy as np
import scipy.io
from keras.utils import np_utils

data = scipy.io.loadmat("data.mat")

all_size = data["seq_tr"].shape[0]

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

import numpy.random
def shuffle_together(a, b):
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

sample_size = 60000
X_train = data["seq_tr"].reshape(sample_size,200,4)
Y_train = data["out_tr"]
X_train_rc = reverse_complement(X_train)
X_train = np.concatenate([X_train,X_train_rc])
Y_train = np.concatenate([Y_train]*2)

X_train,Y_train = shuffle_together(X_train, Y_train)

X_practice = data["seq_te"].reshape(10000,200,4)

X_train = X_train.astype(dtype=np.float32)
Y_train = Y_train.astype(dtype=np.float32)
X_practice = X_practice.astype(dtype=np.float32)

from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Dropout
from keras.layers.convolutional import Convolution1D,ZeroPadding1D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization as BN

np.random.seed(1)

model = Sequential()
model.add(ZeroPadding1D(12,input_shape=(200,4)))
model.add(Dropout(0.1))

#conv_1
model.add(Convolution1D(64, 7,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        subsample_length=2,
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))

model.add(Convolution1D(64, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        subsample_length=2,
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(64, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(64, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(64, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))



#conv_2
model.add(Convolution1D(128, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        subsample_length=2,
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(128, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(128, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(128, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))



#conv_3
model.add(Convolution1D(256, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        subsample_length=2,
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(256, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(256, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(256, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))



#conv_4
model.add(Convolution1D(512, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        subsample_length=2,
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(512, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(512, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Convolution1D(512, 3,
                        border_mode="same",
                        W_constraint=maxnorm(2),
                        activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Flatten())

#fc
model.add(Dense(2048,
                W_constraint=maxnorm(2),
                activation="relu"))
model.add(BN())
model.add(Dropout(0.5))
model.add(Dense(2048,
                W_constraint=maxnorm(2),
                activation="relu"))
model.add(BN())
model.add(Dropout(0.5))


#out
model.add(Dense(8,  
                W_constraint=maxnorm(2),
                activation="sigmoid"))


from keras.optimizers import Adam,SGD
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy","mae"])

from numpy import vstack
from sklearn.metrics import roc_auc_score
import sys

per_epoch = 1
validation_ratio = 0.05
train_size = int(sample_size*2*(1-validation_ratio))

for i in range(1):
  model.fit(X_train,
            Y_train,
            shuffle=True,
            nb_epoch=per_epoch,
            batch_size=500,
            verbose=1,
            validation_split=validation_ratio)
  

  Y_train_pred = model.predict_proba(X_train, verbose=0)
  train_auc = []
  validation_auc = []
  for j in range(8):
    train_auc.append(roc_auc_score(Y_train[:train_size,j],
                                   Y_train_pred[:train_size,j]))
    validation_auc.append(roc_auc_score(Y_train[train_size:,j],
                                        Y_train_pred[train_size:,j]))
  print "train_auc", " ".join(map(lambda f: "%0.03f"%(f,), train_auc))
  print "mean_train_auc", sum(train_auc)/float(len(train_auc))
  print "validation_auc", " ".join(map(lambda f: "%0.03f"%(f,), validation_auc))
  print "mean_validation_auc", sum(validation_auc)/float(len(validation_auc))

  model.save(str((i+1)*per_epoch)+"_epoch_"+"model.hdf")

  Y_practice = model.predict_proba(X_practice, verbose=0)
  res_f = open(str((i+1)*per_epoch)+"_epoch_"+"result.txt","w")
  for v in Y_practice:
    print >> res_f, " ".join(map(str,list(v)))
  sys.stdout.flush()
