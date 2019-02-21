import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class SantanderDataObject(object):
    def __init__(self,batch_size=20000,val_size=0.1):
        self.batch_size=batch_size
        DATA_DIR="c:\\Users\\tihor\\Documents\\kaggle\\santander\\"
        #DATA_DIR = "d:\\\\kaggle\\santander\\"
        df_train = pd.read_csv(DATA_DIR + 'train.csv')
        df_test = pd.read_csv(DATA_DIR + 'test.csv')
        train_y = df_train['target'].values
        train_y=np.expand_dims(train_y,1)
        train_X = df_train[[col for col in df_train.columns if col not in ['ID_code', 'target']]].values
        self.X_train,self.X_val,self.y_train,self.y_val=train_test_split(train_X,train_y,test_size=val_size)
        self.test_ids=df_test['ID_code'].values
        self.X_test=df_test[[col for col in df_test.columns if col not in ['ID_code', 'target']]].values

    def generate_one_epoch(self):
        self.X_train,self.y_train=shuffle(self.X_train,self.y_train)
        num_batches=int(self.X_train.shape[0]//self.batch_size)
        if self.batch_size*num_batches<self.X_train.shape[0]: num_batches+=1
        for i in range(num_batches):
            yield self.X_train[i*self.batch_size:(i+1)*self.batch_size],self.y_train[i*self.batch_size:(i+1)*self.batch_size]

    def generate_dev_data(self):
        num_batches=int(self.X_val.shape[0]//self.batch_size)
        if self.batch_size*num_batches<self.X_val.shape[0]: num_batches+=1
        for i in range(num_batches):
            yield self.X_val[i*self.batch_size:(i+1)*self.batch_size],self.y_val[i*self.batch_size:(i+1)*self.batch_size]

    def generate_test_data(self):
        num_batches=int(self.X_test.shape[0]//self.batch_size)
        if self.batch_size*num_batches<self.X_test.shape[0]: num_batches+=1
        for i in range(num_batches):
            yield self.X_test[i * self.batch_size:(i+1)*self.batch_size]
