#%%
import os
import numpy as np
import pandas as pd


class prep():
    def __init__(self, fdir) -> None:
        super(prep, self).__init__()
        self.fdir = fdir
        self.nv = 3
    
    def train_test_sp(self, df):
        df_test = df.sample(frac=0.2, random_state=24)
        self.index_test = df_test.index
        df_train = df.drop(df_test.index)
        self.index_train = df_train.index
        return (df_train, df_test)

    def load(self, fname):
        df = np.load(self.fdir, allow_pickle=True)
        return df
    
    def scale(self, scaler):
        df_train, df_test = self.train_test_sp(self.load())
        
        train = scaler(df_train)
        test = scaler(df_test)
        return (train, test)
    
    def minmaxscaler(self, df):
        X = df.values[:, :self.nv]
        y = df.values[:, -1]

        a = np.max(X, axis=0) - np.min(X, axis=0)
        b = np.max(y, axis=0) - np.min(y, axis=0)
        ma = np.min(X, axis=0)
        mb = np.min(y, axis=0)

        X = (X - ma) / a
        y = (y - mb) / b

        return (X, y)
    
    def normalscaler(self, df):
        X = df.values[:, 1:self.nv + 1]
        y = df.values[:, -1]

        ma = np.mean(X, axis = 0)
        mb = np.mean(y, axis = 0)
        
        a = np.std(X, axis = 0) 
        b = np.std(y, axis = 0)

        X = (X - ma) / a     
        y = (y - mb) / b

        return (X, y)
    
    def load_grid(self, gdir):
        data = pd.read_csv(gdir, index_col=False)
        data_train = data[self.index_train]
        data_val = data[self.index_test]
        
        self.meean = data_train.mean(0)
        self.std = data_train.std()
        
        train = (data_train - self.mean) / self.std
        test = (data_val - self.meean) / self.std

        return data_train, data_val



#%%



