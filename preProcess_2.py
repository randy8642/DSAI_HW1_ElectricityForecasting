#%% Packages
import numpy as np
import pandas as pd
import os

#%% Arg.
def _PreProcess(fname):

    #%% Path
    fpath = "./data"
    #fname = "training_data.csv"
    H = 30
    #%% Functions
    def _pack(data, H):
        D = []
        for i in range(data.shape[0]):
            if i+H > data.shape[0]:
                break
            else:
                D.append(data[i:i+H, :])
                
        D = np.array(D)
        return D
    
    def _norData(data):
        y = np.copy(data)
        min_MW = np.min(data, axis=0)
        max_MW = np.max(data, axis=0)
        y = (y-min_MW) / (max_MW-min_MW)
    
        return y
    
    def _norLabel(label):
        y = np.copy(label)
        mu = np.mean(y)
        std = np.std(y)
        y = (label-mu) / std
        return y, mu, std
    
    #%% Read csv
    Data = np.array(pd.read_csv(os.path.join(fpath, fname)))
    train_un_data = Data[:769, 4:]
    train_un_label = Data[49:789, 4]
    val_un_data = Data[734:770, 4:]
    val_un_label = Data[789:796, 4]
    test_un_data = Data[762:798, 4:]
    
    train_nor_data = _norData(train_un_data)
    val_nor_data = _norData(val_un_data)
    test_nor_data = _norData(test_un_data)
    
    train_nor_label, mu, std = _norLabel(train_un_label)
    val_nor_label, _, _ = _norLabel(val_un_label)
        
    train_data = _pack(train_nor_data, H)
    val_data = _pack(val_nor_data, H)
    test_data = _pack(test_nor_data, H)
    
    return train_data, train_un_label, val_data, val_un_label, test_data, mu, std

#%% Test
if __name__ == '__main__':
    fname = "training_data.csv"
    train_data, train_nor_label, val_data, val_nor_label, test_data, mu, std = _PreProcess(fname)
