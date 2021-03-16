#%% Packages
import numpy as np
import pandas as pd
import os

#%% Arg.
def _PreProcess(fname):
    #%% Path
    fpath = "./data"
    #fname = "training_data2.csv"
    H = 10
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
    
    #%% Read csv
    Data = np.array(pd.read_csv(os.path.join(fpath, fname)))
    
    # Data
    train_data = {
        '0' : _pack(Data[:110, 4:], H),
        '1' : _pack(Data[117:226, 4:], H),
        '2' : _pack(Data[233:343, 4:], H),
        '3' : _pack(Data[350:460, 4:], H),
        '4' : _pack(Data[467:577, 4:], H),
        '5' : _pack(Data[584:694, 4:], H),
        '6' : _pack(Data[701:811, 4:], H)
        }
    
    train_label = {
        '0' : Data[12:113, 4],
        '1' : Data[129:229, 4],
        '2' : Data[245:346, 4],
        '3' : Data[362:463, 4],
        '4' : Data[479:580, 4],
        '5' : Data[596:697, 4],
        '6' : Data[713:814, 4]    
        }
    
    val_data = {
        '0' : _pack(Data[101:111, 4:], H),
        '1' : _pack(Data[217:227, 4:], H),
        '2' : _pack(Data[334:344, 4:], H),
        '3' : _pack(Data[451:461, 4:], H),
        '4' : _pack(Data[568:578, 4:], H),
        '5' : _pack(Data[685:695, 4:], H),
        '6' : _pack(Data[802:812, 4:], H)    
        }
    
    val_label = {
        '0' : Data[113, 4],
        '1' : Data[229, 4],
        '2' : Data[346, 4],
        '3' : Data[463, 4],
        '4' : Data[580, 4],
        '5' : Data[697, 4],
        '6' : Data[814, 4]    
        }
    
    test_data = {
        '0' : _pack(Data[104:114, 4:], H),
        '1' : _pack(Data[220:230, 4:], H),
        '2' : _pack(Data[337:347, 4:], H),
        '3' : _pack(Data[454:464, 4:], H),
        '4' : _pack(Data[571:581, 4:], H),
        '5' : _pack(Data[688:698, 4:], H),
        '6' : _pack(Data[805:815, 4:], H)    
        }

    return train_data, train_label, val_data, val_label, test_data

#%% Test
if __name__ == '__main__':
    fname = "training_data.csv"
    train_data, train_nor_label, val_data, val_nor_label, test_data = _PreProcess(fname)

