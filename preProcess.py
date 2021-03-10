#%% Packages
import numpy as np
import pandas as pd
import os

#%% Arg.
def _PreProcess(fname):

    #%% Path
    fpath = "./data"
    #fname = "training_data.csv"
    
    #%% Parameters
    H = 60
    
    #%% Functions
    def _pack(data, label, H):
        D = []
        L = []
        for i in range(data.shape[0]):
            if i+H > data.shape[0]:
                break
            else:
                D.append(data[i:i+H, :])
                L.append(label[i:i+H])
                
        D = np.array(D)
        L = np.array(L)
        return D, L
    
    def _nor(x):
        y = np.copy(x)
        for i in range(x.shape[0]):
            min_x = np.min(x[i,:,:])
            max_x = np.max(x[i,:,:])
            nor = (x[i,:,:]-min_x) / (max_x-min_x)
            y[i,:,:] = nor
        return y

            
    
    #%% Read csv
    Data = np.array(pd.read_csv(os.path.join(fpath, fname)))
    train_data = Data[:701, 4:8]
    train_label = Data[60:761, 4]
    test_data = Data[701:761, 4:8].reshape([1,60,4])
    val_label = Data[761:797, 4]
    tra_D_p, tra_L_p = _pack(train_data, train_label, H)
    train_data_nor, test_data_nor = _nor(tra_D_p), _nor(test_data)
    
    return tra_D_p, tra_L_p, val_label, test_data

#%% Test
if __name__ == '__main__':
    fname = "training_data.csv"
    train_data_nor, tra_L_p, val_label, test_data_nor = _PreProcess(fname)
