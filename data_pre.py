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



    




def pre_tcga(tcga_id,tpm,W):
    st="."
    bar_len = 50  #notice! this barcode length can be changed, I dont know how long is the best
    
    k,c0,c1,index_zero,index_one,label=index(tcga_id)
    
    #final_data=np.empty((num,1,8526,600),dtype="float32")
    data_ones=np.empty((c1,1,8526,6*bar_len),dtype="float32")
    data_zeros=np.empty((c0,1,8526,6*bar_len),dtype="float32")
    o_l=0
    z_l=0
    for i in range(0,(tpm.shape[0])): #(0,8525)
        if i in index_zero or i in index_one:
            temp=np.zeros((W.shape[0],W.shape[1]))  #(8526,6)
            data=np.zeros((W.shape[0],W.shape[1]*bar_len))   #(8526,600)
        
            for p in range(0,(W.shape[1])):
                for q in range(0,(W.shape[0])):
                    temp[q,p]=tpm[i,q]*W[q,p]
                
            for y in range(0,(W.shape[0])):
                for z in range(0,6):
                    data[y,z*bar_len:(z+1)*bar_len]=temp[y,z]
        
        
            if label[i]==1:
                data_ones[o_l]=data.reshape([1,8526,6*bar_len])
                o_l = o_l + 1
            if label[i]==0:
                data_zeros[z_l]=data.reshape([1,8526,6*bar_len])
                z_l = z_l + 1
    
        #np.savetxt(file_name, data, fmt='%s', delimiter=' ')
    if k==1 or k==0:
        temp_data=np.concatenate((data_ones,data_zeros),axis=0)
        temp_label=np.concatenate((np.ones(data_ones.shape[0]),np.zeros(data_zeros.shape[0])),axis=0)
    
    elif k > 1:
        temp_data=np.concatenate((data_ones,data_ones),axis=0)
        temp_label=np.concatenate((np.ones(data_ones.shape[0]),np.ones(data_ones.shape[0])),axis=0)
        
        z_all = [i for i in range(data_zeros.shape[0])]
        z_n = len(temp_label)
        z_index = random.sample(z_all,z_n)
        
        temp_zeros=data_zeros[z_index]
        
        temp_data=np.concatenate((temp_data,temp_zeros),axis=0)
        temp_label=np.concatenate((temp_label,np.zeros(temp_zeros.shape[0])),axis=0)
    import pdb;pdb.set_trace()
    arr = np.arange(temp_data.shape[0])
    np.random.shuffle(k)
    final_data = temp_data[arr]
    final_label = temp_label[arr]
    st='.'
    
    
    counts=len(final_label)
    seq_d=(tcga_id,"tpm","npy")
    data_name=st.join(seq_d)
    seq_l=(tcga_id,"label","npy")
    label_name=st.join(seq_l)
    np.save(data_name,final_data)
    np.save(label_name,final_label)
    
    return counts


def index(tcga_id):
    
    st="."
    temp_f=st.join((tcga_id,'index','txt'))
    temp_index=pd.read_csv(temp_f,sep='\t',header=None)
    
    index_l=list(temp_index[0])
    label_l=list(temp_index[1])
    
    llll = [i for i in range(len(label_l))]
    one_index=[]
    zero_index=[]
    for i in range(len(label_l)):
        if label_l[i]==1 and index_l[i]==1:
            one_index.append(i)
        elif label_l[i]==0 and index_l[i]==1:
            zero_index.append(i)
    W0=len(zero_index)
    W1=len(one_index)
    k=W0//W1
    
    
    #temp_count=200-W1
    #select_zero_index=random.sample(zero_)
    return k,W0,W1,zero_index,one_index,label_l



def data_pre(case_lists,tpm_names,label_names,w):
    counts=np.arange(0,len(case_lists))
    for i in range(0,len(case_lists)):
        case=case_lists[i]
        print(case)
        tpm=np.loadtxt(tpm_names[i],dtype=np.float,delimiter="\t")
        #label=np.loadtxt(label_names[i],dtype=np.int,delimiter="\t")
        counts[i]=pre_tcga(case,tpm,w)
    
    return counts
    

def data_merge(case_list):
    st='.'
    seq_d=(case_list[0],'tpm','npy')
    seq_l=(case_list[0],'label','npy')
    name_d=st.join(seq_d)
    name_l=st.join(seq_l)
    merge_data=np.load(name_d)
    merge_label=np.load(name_l)
    import pdb;pdb.set_trace()
    for i in range(1,len(case_lists)):
        seq_d=(case_list[i],'tpm','npy')
        seq_l=(case_list[i],'label','npy')
        name_d=st.join(seq_d)
        name_l=st.join(seq_l)
        temp_d=np.load(name_d)
        temp_l=np.load(name_l)
        merge_data=np.concatenate((merge_data,temp_d),axis=0)
        merge_label=np.concatenate((merge_label,temp_l),axis=0)
    import pdb;pdb.set_trace()
    arr = np.arange(merge_data.shape[0])
    np.random.shuffle(arr)
    data = merge_data[arr]
    label = merge_label[arr]
    
    return data.shape,data,label







if __name__ == "__main__":
    
    case_lists=["lusc-bar","brca-bar","skcm-bar","stad-bar","luad-bar","paad-bar"]
    Wgc=np.loadtxt("X_gc_noweight.txt",dtype=np.float,delimiter=" ")
    Wgc=np.matrix(Wgc)
    tpm_names=['LUSC_TPM_feature.txt','BRCA_TPM_feature.txt','SKCM_TPM_feature.txt','STAD_TPM_feature.txt','LUAD_TPM_feature.txt','PAAD_TPM_feature.txt']
    label_names=['LUSC_label.txt','BRCA_label.txt','SKCM_label.txt','STAD_label.txt','LUAD_label.txt','PAAD_label.txt']
    counts=[0,0,0,0,0,0]
    
    
    if os.path.exists('brca-bar.tpm.npy')==False:
        data_pre(case_lists,tpm_names,label_names,Wgc) 
    
    


