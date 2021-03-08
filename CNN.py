import numpy as np
import pandas as pd
import os
import random
import math
#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os.path
from sklearn.model_selection import KFold
import tensorflow as tf
import keras

from keras import layers
from sklearn.metrics import roc_curve, auc, average_precision_score
import keras.backend as K
#import keras.backend.tensorflow_backend as KTF

import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.layers import Add,ZeroPadding2D,BatchNormalization,LeakyReLU,GlobalAveragePooling2D,MaxPool2D
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization,Flatten
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping,LearningRateScheduler
from sklearn.metrics import roc_auc_score
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
import math

#from keras.utils import multi_gpu_model
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
#from sklearn.svm import SVC
#from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 #half of the memory
#set_session(tf.compat.v1.Session(config=config))
#config = tf.ConfigProto()

#sess = tf.Session(config=config)
#KTF.set_session(sess)
#from YX_model import model_train , creatcnn # A bettet model required
event_num=2
seed = 1
CV=10
#vector_size = 572
droprate = 0.3

learning_rate = 0.02
num_epoches = 10




def resblock_body(x,num_filters,num_blocks):
        #x = ZeroPadding2D(((1,0),(1,0)))(x)
    for i in range(num_blocks):
        y = Conv2D( num_filters, (3,1), padding = 'same')(x)
        y = LeakyReLU(alpha=0.1)(y)
        y = Conv2D(num_filters, (3,1), padding = 'same')(y)
        y = LeakyReLU(alpha=0.1)(y)
        x = Add()([x,y])
    return x


def CNN5():
    #train_input = Input(shape=(vector_size * 2,), name='Inputlayer')
    model = Sequential()
    
    x_input = Input((300,1,8526))
    x = Conv2D(128,(3,3),(2,2),padding='same')(x_input)
    x = LeakyReLU(alpha=0.1)(x)
    x = resblock_body(x,128,1)
    x = Conv2D(64,kernel_size =(3,3),strides = (2,2), padding='same')(x_input)
    x = LeakyReLU(alpha=0.1)(x)
    x = resblock_body(x,64,1)
    x = Conv2D(64,kernel_size=(3,3),strides = (2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    x = Dense(16,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(droprate)(x)
    x = Dense(event_num,activation='softmax')(x)
    model = Model(x_input,x,name='CNN-DDI')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    #import pdb;pdb.set_trace()
    return model


def data_load():
    data=np.load("data_1.npy")
    label=np.load("label_1.npy")
    shape=data.shape
    print(shape)
    #import pdb;pdb.set_trace()
    data=data.reshape([shape[0],shape[3],shape[1],shape[2]])
    
    return shape,data,label

def get_index(label_matrix,event_num, seed,CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix==j)
        kf = KFold(n_splits=CV, shuffle=True, random_state = seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            index_all_class[index[0][test_index]] = k_num
            k_num+=1
    return index_all_class
    


def train_test(data,label,ratio):
    print("@@@@@@@@@@@@@@@@@\n")
    
    data_temp, data_test, label_temp, label_test = train_test_split(data, label, test_size=(ratio),random_state=1)
    #import pdb;pdb.set_trace()
    return data_temp, data_test, label_temp, label_test


def select_data_cv(data,label,cv):
    
    #data_temp, data_test, label_temp, label_test = train_test_split(data, label, test_size=(1-ratio),random_state=1)
    
    k=data.shape[0]//10
    cv=cv
    
    if k*(cv+1)<=data.shape[0]:
        seq=np.arange(k*cv,k*(cv+1))
        data_val=data[k*cv:(k*(cv+1)-1)]
        label_val=label[k*cv:(k*(cv+1)-1)]
        
        data_train=np.delete(data,seq,axis=0)
        label_train=np.delete(label,seq,axis=0)
    elif k*(cv+1)>data.shape[0]:
        seq=np.arange(k*cv,data.shape[0])
        data_val=data[k*cv:data.shape[0]-1]
        label_val=label[k*cv:data.shape[0]-1]
        data_train=np.delete(data,seq,axis=0)
        label_train=np.delete(label,seq,axis=0)
    
    return data_train,label_train,data_val,label_val
    


if __name__ == "__main__":
    
    
    
    shape,data,label=data_load()
    
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0,event_num),dtype=float)
    
    index_all_class= get_index(label,event_num,seed,CV)
    import pdb;pdb.set_trace()
    for k in range(0,10): # 
        print("______________k___________",k)
        train_index = np.where(index_all_class != k)
        test_index = np.where(index_all_class == k)
        pred = np.zeros((len(test_index[0]),event_num),dtype=float)
        x_train = data[train_index]
        x_test  = data[test_index]
        
        y_train = label[train_index]
        y_test  = label[test_index]
        y_train_one_hot = np.array(y_train)
        y_train_one_hot = (np.arange(y_train_one_hot.max()+1)==y_train[:,None]).astype(dtype='float32')
        
        y_test_one_hot = np.array(y_test)
        y_test_one_hot = (np.arange(y_test_one_hot.max()+1)==y_test[:,None]).astype(dtype='float32')

              
        print("!!!!!!!!!!!!\n")
        
        cnn = CNN5()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        def scheduler(epoch):
            if epoch<10:
                K.set_value(cnn.optimizer.lr,0.02)
                return K.get_value(cnn.optimizer.lr)
            elif epoch<15:
                K.set_value(cnn.optimizer.lr,0.002)
                return K.get_value(cnn.optimizer.lr)
            elif epoch<25:
                K.set_value(cnn.optimizer.lr,0.0002)
                return K.get_value(cnn.optimizer.lr)
            else :
                K.set_value(cnn.optimizer.lr,0.00002)
                return K.get_value(cnn.optimizer.lr)
            
        reduce_lr = LearningRateScheduler(scheduler)
        cnn.fit(x_train, y_train_one_hot, batch_size=32, epochs=28, validation_data=(x_test, y_test_one_hot),
                        callbacks=[reduce_lr])
        pred += cnn.predict(x_test)
        pred_score = pred / 1
        pred_type = np.argmax(pred_score, axis=1)
        y_true = np.hstack((y_true, y_test))
        y_pred = np.hstack((y_pred, pred_type))
        
        
        frawt=np.save((str(k)+'.ytrue'),y_true)
        fyp=np.save((str(k)+'.ytype'),y_true)
        frawp=np.save((str(k)+'.ypred'),pred)
        
        import pdb;pdb.set_trace()
        
        
        
        print("___________________test end_________\n")
    



