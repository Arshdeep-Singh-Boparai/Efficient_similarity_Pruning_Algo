#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 21:50:45 2021

@author: arshdeep
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:35:36 2021

@author: arshdeep
"""
import h5py    
import numpy as np    
#f1 = h5py.File('/home/arshdeep/DCASE2021_task1a/model_quant_fold_1.h5','r+')   

import tensorflow as tf
from sklearn.metrics import classification_report,confusion_matrix

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, BatchNormalization,Flatten, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (BatchNormalization, SeparableConv2D, Conv2D,MaxPooling2D,  ZeroPadding2D,AveragePooling2D, Activation, Flatten, Dropout, Dense)
from tensorflow.keras import backend as K


from tensorflow.keras.callbacks import ModelCheckpoint
import os
from numpy import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.keras.models import load_model
from sklearn.metrics import log_loss



#%% data load

x_train=np.load('~/train_data_new.npy')#.astype(np.float32)

#x_train = np.reshape(X_train, [6108*19,96,64,1])


x_test=np.load('~/test_data.npy')#.astype(np.float32)

labels_test=np.load('~/labels_test.npy')
labels_train=np.load('~/labels_train.npy')

# y_train=np.vstack((Y_train,y_valid))
C = tf.constant(10, name = "C")
      
one_hot_matrix_test = tf.one_hot(labels_test, C, on_value = 1.0, off_value = 0.0, axis =-1)
      
one_hot_matrix_train = tf.one_hot(labels_train, C, on_value = 1.0, off_value = 0.0, axis =-1)  
sess = tf.Session()

y_test = sess.run(one_hot_matrix_test)
y_train = sess.run(one_hot_matrix_train) 
sess.close()


#print(np.shape(x_train),np.shape(x_test))

X_test = np.reshape(x_test, [2518*19,96,64,1])

#%%

W=list(np.load('~/Baseline_65/best_weights_VGGish.npy',allow_pickle=True)) # download from the given links.




print('.......... Cosine  Pruning............')
#os.chdir('/home/arshdeep/Pruning/SPL/VGGish_applied_acoustics/important_index/')

L0=sorted(np.load('~/VGGish/important_index/COSINE_SIM/sim_index0.npy'))#[64-len(L0):64])
L2=sorted(np.load('~/VGGish/important_index/COSINE_SIM/sim_index2.npy'))#[128-len(L2):128])#np.arange(0,64)#
L4=sorted(np.load('~/VGGish/important_index/COSINE_SIM/sim_index4.npy'))#[256-len(L4):256])
L6=sorted(np.load('~/VGGish/important_index/COSINE_SIM/sim_index6.npy'))#[256-len(L6):256])
L8=sorted(np.load('~/VGGish/important_index/COSINE_SIM/sim_index8.npy'))#[512-len(L8):512])#np.arange(0,64)#
L10=sorted(np.load('~/VGGish/important_index/COSINE_SIM/sim_index10.npy'))#[512-len(L10):512])
#np.save('Baseline_weights',W_dcas)

D3=L10  #list of indexes to be removed from the dense layer

w_f=[]
for i in range(len(L10)):
	w_f.append(list(range(D3[i]*24,D3[i]*24+24)))
	
w_f=np.hstack(w_f)
Total_filter=len(L0)+len(L2)+len(L4)+len(L6)+len(L8)+len(L10)#+len(L7)+len(L8)+len(L9)+len(L10)+len(L11)+len(L12)+len(L13)





W_pruned=[W[0][:,:,:,L0],W[1][L0],W[2][:,:,L0,:][:,:,:,L2],W[3][L2],W[4][:,:,L2,:][:,:,:,L4],W[5][L4],	W[6][:,:,L4,:][:,:,:,L6],W[7][L6],W[8][:,:,L6,:][:,:,:,L8],W[9][L8],W[10][:,:,L8,:][:,:,:,L10],W[11][L10],W[12][w_f,:],W[13],W[14],W[15],W[16],W[17]]


#%% Architecture load



input_shape = (96, 64, 1)
aud_input = Input(shape=input_shape, name='input_1')

# Block 1
x = Conv2D(len(L0), (3, 3), strides=(1, 1), activation='relu', kernel_regularizer='l2', padding='same', name='conv1')(aud_input)

#x=Dropout(0.5)(x)


x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

# Block 2
x = Conv2D(len(L2), (3, 3), strides=(1, 1), activation='relu',kernel_regularizer='l2',  padding='same', name='conv2')(x)

#x =Dropout(0.5)(x)


x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

# Block 3
x = Conv2D(len(L4), (3, 3), strides=(1, 1), activation='relu', kernel_regularizer='l2', padding='same', name='conv3/conv3_1')(x)
#x =Dropout(0.5)(x)



x = Conv2D(len(L6), (3, 3), strides=(1, 1), activation='relu', kernel_regularizer='l2', padding='same', name='conv3/conv3_2')(x)

#x =Dropout(0.5)(x)


x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

# Block 4
x = Conv2D(len(L8), (3, 3), strides=(1, 1), activation='relu',kernel_regularizer='l2',  padding='same', name='conv4/conv4_1')(x)

#x =Dropout(0.5)(x)

x = Conv2D(len(L10), (3, 3), strides=(1, 1), activation='relu', kernel_regularizer='l2', padding='same', name='conv4/conv4_2')(x)

#x =Dropout(0.5)(x)


x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)



model = Model(aud_input, x, name='VGGish_dens1')

#model.set_weights(W_pruned[0:12])  

x = Flatten(name='flatten_')(x)
x = Dense(4096, activation='relu',kernel_regularizer='l2', name='vggish_fc1/fc1_1')(x)


#model = Model(aud_input, x, name='VGGish_dens1')

# model.set_weights(np.load('D:/PhD_data_system/VGGish_pruning/VGGish_dense1_weights.npy'))	


#d1_p =Dropout(0.5)(x)

d1 = Dense(128, activation='relu', kernel_regularizer='l2',name='vggish_dense2')(x)
out = Dense(10, activation='softmax', name='vggish_out')(d1)

model_all=Model(aud_input,out,name='classification')

model_all.set_weights(W_pruned)

model_all.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


#%%

checkpointer = ModelCheckpoint(filepath='~/best_weights_VGGish_1.h5py',monitor='val_acc',verbose=1, save_best_only=True,save_weights_only=True)
hist=model_all.fit(x_train, y_train, batch_size=64,epochs=100,verbose=1,validation_data=(X_test, y_test),callbacks=[checkpointer])


model_all.load_weights('~/best_weights_VGGish_1.h5py')

model_all.save('~/best_model_VGGish_1.h5')

np.save('~/history_1.npy',hist.history)
#%% Baseline testing



score=model_all.predict(X_test)
#%%
num_segment=19;
pred_prob=[]
t=0
num=int(np.size(labels_test)/num_segment);
for i in range(num):
    pred=np.sum(score[t:t+num_segment,:],0)
    pred_prob.append(pred)
    t=t+num_segment


pred_prob=np.argmax((np.asarray(pred_prob)),1)

k=0
true_labels=[]
for i in range(num):
	true_labels.append(labels_test[k])
	k=k+num_segment


true_label=np.asarray(true_labels)
asd=confusion_matrix(true_label,pred_prob);
accu=(np.trace(asd)/np.size(true_label))*100;

print(accu,'cosine sim: layerwise C1_6:5')
model_all.summary()
