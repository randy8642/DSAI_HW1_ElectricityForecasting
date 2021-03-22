
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
    train_data = _pack(Data[:750, 5:8], H)
    train_label = _packL(Data[30:758, 4], H=8)
    val_data = _pack(Data[750:780, 5:8], H)
    val_label = _packL(Data[780:788, 4], H=8)
    test_data = _pack(Data[780:810, 5:8], H)

    return train_data, train_label, val_data, val_label, test_data

def _PreProcess2(fname):
    # Path
    fpath = "./data"
    #fname = "training_data.csv"
    # Read csv
    df = pd.read_csv(os.path.join(fpath, fname))
    # Data
    VAL_data = df.loc[:794, :]    # time by data
    TES_data = df

    return VAL_data, TES_data


def _Plot(fname):
    fpath = "./data"
    Data = np.array(pd.read_csv(os.path.join(fpath, fname)))
    MW = Data[:805, 4]
    perc = Data[:805, 5]
    PP_MW = Data[:805, 6]
    P_MW = Data[:805, 7]
    VAL_MW = Data[780:788, 4]

    return MW, perc, PP_MW, P_MW, VAL_MW

#%% Test
if __name__ == '__main__':
    fname = "training_data.csv"
    fname2 = "training_data2.csv"
    train_data, train_label, val_data, val_label, test_data = _PreProcess(fname)
    val_data2, test_data2 = _PreProcess2(fname2)


    print("train_data >>", train_data.shape)
    print("train_label >>", train_label.shape)
    print("val_data >>", val_data.shape)
    print("val_label >>", val_label.shape)
    print("test_data >>", test_data.shape)
    print("====================")
    print("val_data >>", val_data2.shape)
    print("test_data >>", test_data2.shape)    