# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 01:26:16 2021

@author: Sean
"""

import keras
import datetime
import itertools
import numpy as np
import tensorflow as tf
import sklearn
from sklearn import metrics
from IPython import get_ipython


from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras import backend as K

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib','inline')
import warnings
warnings.filterwarnings('ignore')

mode = 'Testing'

# global parameters
num_features = 43

# model parameters
batch_size = 1024
n_epochs = 300

# Layer classes
# inherited to writout summary for tensorboard
class MyLSTM(LSTM):
    def call(self, inputs, mask=None, training=None, initial_state=None):
        ret = super(MyLSTM, self).call(inputs,mask=mask,training=training,initial_state=initial_state)
        activation = ret
        tf.summary.histogram('activation',activation)
        return activation


class MyDense(Dense):
    def call(self, inputs):
        activation = super(MyDense, self).call(inputs)
        tf.summary.histogram('activation',activation)
        return activation


class Subtract(keras.layers.Layer):
    def __init__(self, value=0.0, **kwargs):
        self.init_value = np.float32(value)
        super(Subtract, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.value = self.add_weight(name="value",shape=(1,1,num_features),initializer=keras.initializers.Constant(value=np.float32(self.init_value)),trainable=False)
        super(Subtract, self).build(input_shape)
        
    def call(self, inputs):
        return inputs - self.value
            
            
class Multiply(keras.layers.Layer):
    def __init__(self, value=1.0, **kwargs):
        self.init_value = np.float32(value)
        super(Multiply, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.log_value = self.add_weight(name="log_value",shape=(1,1,num_features,),initializer=keras.initializers.Constant(value=np.log(np.float32(self.init_value))),trainable=False)
        super(Multiply, self).build(input_shape)
        
    def call(self, inputs):
        return inputs * tf.exp(self.log_value)
    
    
class Conv1D(keras.layers.Conv1D):
    def call(self, inputs):
        activation = super(Conv1D, self).call(inputs)
        tf.summary.histogram('activation',activation)
        return activation
    
    
class NoisyVoting1D(keras.layers.Layer):
    def call(self, inputs):
        random_weights = keras.backend.random_uniform(shape=tf.shape(inputs)[:2],minval=0.5,maxval=1.5)
        random_weights = tf.expand_dims(random_weights, axis=-1)
        random_weight_sum = keras.backend.sum(random_weights,axis=[1],keepdims=True)
        random_weights = random_weights / random_weight_sum
        activation = keras.backend.sum(random_weights * inputs,axis=[1],keepdims=True)
        return activation
    
def load_data(dataset):
    data = np.load(dataset, allow_pickle=True)
    X = np.concatenate(data[:,0])[:, :, :num_features]
    y = np.concatenate(data[:,1])
    n_prog = len(y[y == 1])
    n_nonprog = len(y[y == 0])
    print (f'Number of prog samples = {n_prog}')
    print (f'Number of nonprog samples = {n_nonprog}')
    assert y.shape[0] == X.shape[0]
    return X, y


# Building the model
def NN(input_shape, mean, std):
    model = Sequential()
    model.add(Subtract(mean, input_shape=input_shape))
    model.add(Multiply(1/std))
    model.add(Conv1D(filters=16,kernel_size=1,strides=1,kernel_initializer="he_uniform",activation='elu'))
    model.add(keras.layers.SpatialDropout1D(rate=0.1))
    for layer_idx in range(6):
        model.add(Conv1D(filters=8,kernel_size=5,strides=1,padding="same",kernel_initializer=keras.initializers.VarianceScaling(scale=2.0,mode='fan_in',distribution='uniform',seed=None),activation='elu'))
        model.add(keras.layers.SpatialDropout1D(rate=0.1))
        model.add(keras.layers.MaxPooling1D(pool_size=2,strides=2,padding="valid"))
    
    model.add(Conv1D(filters=1,kernel_size=1,strides=1,dilation_rate=1,padding="same",kernel_initializer=keras.initializers.VarianceScaling(scale=1.0,mode='fan_in',distribution='uniform',seed=None),activation=None))
    model.add(NoisyVoting1D())
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Activation('sigmoid'))
    return model


def train(model, X_train, y_train, X_validation, y_validation):
# tensorboard log out
    now = datetime.datetime.now()
    time_label = now.strftime("%Y%m%d_%H-%M")
    tensorboard = keras.callbacks.TensorBoard(log_dir = f'./CNN-{time_label}' ,histogram_freq=1,batch_size=batch_size,write_graph=True)
    print("Training ...")
    model.fit(X_train,y_train,epochs=n_epochs,batch_size=batch_size,callbacks=[tensorboard],validation_data=(X_validation, y_validation))
    print("\nValidating ...")
    score, accuracy = model.evaluate(X_validation,y_validation,batch_size=batch_size,verbose=1)
    print("Dev loss: ", score)
    print("Dev accuracy: ", accuracy)
    print("Saving the weights ...")
    model.save_weights(f'./saved_models/CNNW-{time_label}.h5')
    
def print_summary_statistics(pred_prob_song,pred_class_song, true_class_song, name_vl_song, im):
    print("\nsong_name, prob_prog(class1), pred_class, true_class, correct?\n")
    for j in range(len(name_vl_song)):
        if pred_class_song[j] == 0:
            pred_label = 'nonprog'
        else:
            pred_label = 'prog '
        if true_class_song[j] == 0:
            true_label = 'nonprog'
        else:
            true_label = 'prog '
        if pred_class_song[j] != true_class_song[j]:
            correct = ' x '
        else:
            correct = 'good!'
        print('{0} {1:5.3f} {2} {3} {4}'.format(name_vl_song[j][:40].ljust(41),pred_prob_song[j],pred_label,true_label,correct))
# plotting song barcode
        if true_class_song[j] == 0:
            period = 12
        elif true_class_song[j] == 1:
            period = 8
        every = 100
        tot_duration = ( period * im[j].shape[1] / 100 ) + 10
        mapping = np.arange(0, im[j].shape[1], every * 100 / period)
        fig, ax = plt.subplots(figsize=(6, 0.25))
        ax.imshow(im[j],#interpolation='nearest',
                  aspect='auto',
                  cmap='binary')
        ax.get_yaxis().set_ticks([])
        plt.xticks(mapping, np.arange(10, tot_duration, every, dtype=int))
        plt.tight_layout()
        plt.show()
    cm = metrics.confusion_matrix(true_class_song,pred_class_song)
    accu = metrics.accuracy_score(true_class_song,pred_class_song)
    info = metrics.balanced_accuracy_score(true_class_song,pred_class_song,adjusted=True)
    print('\nPrediction accuracy:', accu)
    ('Informedness:', info)
    print('\nConfusion Matrix:')
    print(cm)
    return cm,accu,info


def predict(weight_file, dataset, test=False):
    model.load_weights(weight_file)
    labels = []
    probs = []
    names = []
    y_true = []
    im = []
    data = np.load(dataset, allow_pickle=True)
    if test == True:
        temp = list(itertools.chain.from_iterable(data)) #
        data = np.asarray(temp).reshape((int(len(temp)/3),3)) #
    for d in data:
        name = d[2]
        X = d[0]
        y = d[1][0]
        names.append(name)
        y_true.append(y)
        p = model.predict(X) # array of btw 0,1
        prob = np.average(p) # btw 0,1
        # creating array of pixels
        prob_color = np.round(p.reshape((len(d[1]),1)) * 255)
        prob_color = np.repeat(prob_color, 100)
        prob_color = np.tile(prob_color, (1,1))
        im.append(prob_color)
        #im = Image.fromarray(prob_color)
        #im.save(f'{data[2]}.jpg')
        p[p < 0.5] = 0
        p[p >=0.5] = 1
        vote = np.average(p) # btw 0,1
        probs.append(vote)
        if vote == 0.5:
            vote = vote + 0.1
        label = int(round(vote))
        labels.append(label) # 0 or 1
    return labels, probs, names, y_true, im


########################################
### Loading training/validation data ###
########################################
print (f'\nLoading the training set ...')
X_train, y_train =load_data('../data_prep/43feature_proginprog/_train.npy')
print (f'\nLoading validation set ...')
X_validation, y_validation =load_data('../data_prep/43feature_proginprog/_validation.npy')
print(f'\nX_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_validation shape: {X_validation.shape}')
print(f'y_validation shape: {y_validation.shape}')
assert y_train.shape[1:] == y_validation.shape[1:]
mean = np.mean(X_train, axis=(0, 1), keepdims=True)
std = np.std(X_train, axis=(0, 1), keepdims=True)
input_shape = (X_train.shape[1], X_train.shape[2])
########################################
### training/validation ###
########################################
if mode == 'Training':
    model = NN(input_shape, mean, std)
    model.summary()
    print("Compiling ...")
    # optimizer
    adam = Adam(lr=0.001, decay=1.e-7)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    train(model, X_train, y_train, X_validation, y_validation)
########################################
### testing ###
########################################
elif mode == 'Testing':
    try: del model
    except: pass
    model = NN(input_shape, mean, std)
    adam = Adam(lr=0.001, decay=1.e-7)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    print("\n######################################################")
    print("Model summary:")
    print("######################################################")
    model.summary()
    labels, probs, names, y_true, im =predict('./saved_models/CNNW-20190501_16-50.h5','../data_prep/43feature_proginprog/_train.npy')
    print("\n######################################################")
    print("Training set statistics:")
    print("######################################################")
    print("\nA picture worth a thousand words!")
    print_summary_statistics(probs, labels, y_true, names, im)
    score, accuracy = model.evaluate(X_train,y_train,batch_size=batch_size,verbose=0)
    print("Model loss: ", score)
    print("Model accuracy: ", accuracy)
    print("\n######################################################")
    print("Validation set statistics:")
    print("######################################################")
    try: del model
    except: pass
    model = NN(input_shape, mean, std)
    adam = Adam(lr=0.001, decay=1.e-7)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    labels, probs, names, y_true, im =predict('./saved_models/CNNW-20190501_16-50.h5','../data_prep/43feature_proginprog/_validation.npy')
    print_summary_statistics(probs, labels, y_true, names, im)
    score, accuracy = model.evaluate(X_validation,y_validation,batch_size=batch_size,verbose=0)
    print("Model loss: ", score)
    print("Model accuracy: ", accuracy)
    print("\n######################################################")
    print("Test set statistics:")
    print("######################################################")
    try: del model
    except: pass
    model = NN(input_shape, mean, std)
    adam = Adam(lr=0.001, decay=1.e-7)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    labels, probs, names, y_true, im =predict('./saved_models/CNNW-20190501_16-50.h5','../data_prep/43feature_proginprog/_test.npy',test=True)
    print_summary_statistics(probs, labels, y_true, names, im)
    score, accuracy = model.evaluate(X_validation,y_validation,batch_size=batch_size,verbose=0)
    print("Model loss: ", score)
    print("Model accuracy: ", accuracy)
    print("\n######################################################")
    print("Testing on djent:")
    print("######################################################")
    try: del model
    except: pass
    model = NN(input_shape, mean, std)
    adam = Adam(lr=0.001, decay=1.e-7)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    labels, probs, names, y_true, im =predict('./saved_models/CNNW-20190501_16-50.h5','../data_prep/43feature_proginprog/_djent.npy',test=True)
    print_summary_statistics(probs, labels, y_true, names, im)

