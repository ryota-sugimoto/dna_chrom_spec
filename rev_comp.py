#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat

data = loadmat("data.mat")
train_seq = data["seq_tr"].reshape(60000,200,4)

comp_mat = np.matrix([[0,0,0,1],
                      [0,0,1,0],
                      [0,1,0,0],
                      [1,0,0,0]])

l = []
for seq in train_seq:
  a = np.array(seq*comp_mat)
  a = a[::-1]
  l.append(a)

rc = np.array(l)
print rc.shape
