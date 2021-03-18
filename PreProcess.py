#%% Packages
import numpy as np
import pandas as pd
import os

#%% Arg.
def _PreProcess(fname):
    # Path
    fpath = "./data"
    #fname = "training_data.csv"
    H = 30
    # Functions
    def _pack(data, H):
        D = []
        for i in range(data.shape[0]):
            if i+H > data.shape[0]:
                break
            else:
                D.append(data[i:i+H, :])

        D = np.array(D)
        return D
    
    def _packL(label, H):
        D = []
        for i in range(label.shape[0]):
            if i+H > label.shape[0]:
                break
            else:
                D.append(label[i:i+H])

        D = np.array(D)
        return D    

    # Read csv
    Data = np.array(pd.read_csv(os.path.join(fpath, fname)))

    # Data
    train_data = _pack(Data[:777, 5:8], H)
    train_label = _packL(Data[30:791, 4], H=14)
    val_data = _pack(Data[761:791, 5:8], H)
    val_label = _packL(Data[791:805, 4], H=14)
    test_data = _pack(Data[775:805, 5:8], H)

    return train_data, train_label, val_data, val_label, test_data

def _Plot(fname):
    fpath = "./data"
    Data = np.array(pd.read_csv(os.path.join(fpath, fname)))
    MW = Data[:806, 4]
    perc = Data[:806, 5]
    PP_MW = Data[:806, 6]
    P_MW = Data[:806, 7]
    VAL_MW = Data[791:805, 4]
    
    return MW, perc, PP_MW, P_MW, VAL_MW

#%% Test
if __name__ == '__main__':
    fname = "training_data.csv"
    train_data, train_label, val_data, val_label, test_data = _PreProcess(fname)

