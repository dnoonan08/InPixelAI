from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np

import itertools

groups2x2 = []
for i in range(71):
    for j in range(71):
        
        groups2x2.append([i*72+j,     i*72+(j+1),
                          (i+1)*72+j, (i+1)*72+(j+1)])

Remap_2x2 = np.zeros((72*72,len(groups2x2)))
for isc,sc in enumerate(groups2x2): 
    for tc in sc:
        Remap_2x2[int(tc),isc]=1
tf_Remap_2x2 = tf.constant(Remap_2x2,dtype=tf.float32)

groups4x4 = []
for i in range(69):
    for j in range(69):
        
        groups4x4.append([i*72+j,     i*72+(j+1),     i*72+(j+2),     i*72+(j+3),
                          (i+1)*72+j, (i+1)*72+(j+1), (i+1)*72+(j+2), (i+1)*72+(j+3),
                          (i+2)*72+j, (i+2)*72+(j+1), (i+2)*72+(j+2), (i+2)*72+(j+3),
                          (i+3)*72+j, (i+3)*72+(j+1), (i+3)*72+(j+2), (i+3)*72+(j+3)])

Remap_4x4 = np.zeros((72*72,len(groups4x4)))
for isc,sc in enumerate(groups4x4): 
    for tc in sc:
        Remap_4x4[int(tc),isc]=1
tf_Remap_4x4 = tf.constant(Remap_4x4,dtype=tf.float32)


def telescopeMSE(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)

    # pixel-level MSE
    lossMSE1 = K.mean(K.square(y_true - y_pred), axis=(-1))
    
    # 2x2 pixel group level MSE
    y_pred_2 = tf.matmul(y_pred, tf_Remap_2x2)/4.
    y_true_2 = tf.matmul(y_true, tf_Remap_2x2)/4.
    lossMSE2 = K.mean(K.square(y_true_2 - y_pred_2), axis=(-1))

    # 4x4 pixel group level MSE
    y_pred_4 = tf.matmul(y_pred, tf_Remap_4x4)/16.
    y_true_4 = tf.matmul(y_true, tf_Remap_4x4)/16.
    lossMSE4 = K.mean(K.square(y_true_4 - y_pred_4), axis=(-1))

    return 4*lossMSE1 + 2*lossMSE2 + lossMSE4


def sqrt_mse(y_true, y_pred):
    y_true_sqrt = K.sqrt(K.sqrt(K.square(K.cast(y_true, y_pred.dtype))))
    y_pred_sqrt = K.sqrt(K.sqrt(K.square(y_pred)))
    loss   = K.mean(K.square(y_true_sqrt - y_pred_sqrt),axis=(-1))
    return loss



r  = ((np.array(list(itertools.product(np.arange(72),np.arange(72)))) - [36,36])**2).sum(axis=1)**.5
_weight = r-8
_weight[_weight<1] = 1

def r_weighted_mse(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    weight = K.cast(_weight,y_pred.dtype)
    loss   = K.mean(K.square(y_true - y_pred)*weight,axis=(-1))
    return loss

def r2_weighted_mse(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    weight = K.cast(_weight,y_pred.dtype)
    loss   = K.mean(K.square(y_true - y_pred)*weight*weight,axis=(-1))
    return loss
    
def r3_weighted_mse(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    weight = K.cast(_weight,y_pred.dtype)
    loss   = K.mean(K.square(y_true - y_pred)*weight*weight*weight,axis=(-1))
    return loss
    
r  = ((np.array(list(itertools.product(np.arange(72),np.arange(72)))) - [36,36])**2).sum(axis=1)**.5
_weightMask = np.where(r<8,1,10)

def rMask_weighted_mse(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    weight = K.cast(_weightMask,y_pred.dtype)
    loss   = K.mean(K.square(y_true - y_pred)*weight,axis=(-1))
    return loss

def rMask2_weighted_mse(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    weight = K.cast(_weightMask,y_pred.dtype)
    loss   = K.mean(K.square(y_true - y_pred)*weight*weight,axis=(-1))
    return loss

def rMask3_weighted_mse(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    weight = K.cast(_weightMask,y_pred.dtype)
    loss   = K.mean(K.square(y_true - y_pred)*weight*weight*weight,axis=(-1))
    return loss

def mseTest(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    loss   = K.mean(K.square(y_true - y_pred),axis=(-1))
    return loss

def InvWeightedMSE(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    weight = K.maximum(y_pred,.5,axis=(-1))
    loss   = K.mean(K.square((y_true - y_pred)/weight),axis=(-1))
    return loss

