#!/usr/bin/env python

import os
import struct
import numpy as np
import scipy.io

data = scipy.io.loadmat("data.mat")

all_size = data["seq_tr"].shape[0]
train_size = 50000

X_train = data["seq_tr"][:train_size].reshape(train_size,200,4)
Y_train = data["out_tr"][:train_size]

X_test = data["seq_tr"][train_size:].reshape(all_size-train_size,200,4)
Y_test = data["out_tr"][train_size:]

import theano
theano.config.floatX = "float32"
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

from keras.models import Sequential
from keras.layers.core import Dense,Flatten,Dropout
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD

np.random.seed(32426242)

model = Sequential()
model.add(Convolution1D(32, 5,
                        border_mode="valid",
                        input_dim=4,
                        input_length=200,
                        subsample_length=2,
                        activation="relu"))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(8, activation="sigmoid"))
model.compile(loss="categorical_crossentropy",
              optimizer="adadelta",
              metrics=["accuracy"])

model.fit(X_train,
          Y_train,
          nb_epoch=50,
          batch_size=300,
          verbose=1,
          validation_split=0.1)

Y_train_pred = model.predict_classes(X_train, verbose=0)
train_acc = np.sum(Y_train == Y_train_pred, axis=0) / float(X_train.shape[0])
print "train_acc", train_acc

Y_test_pred = model.predict_classes(X_test, verbose=0)
test_acc = np.sum(Y_test == Y_test_pred, axis=0) / float(X_test.shape[0])
print "test_acc", test_acc
