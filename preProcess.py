if __name__ == '__main__':
    #%% Packages
    import argparse
    import numpy as np
    import pandas as pd
    import os
    
    #%% Arg.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        default='data_set.csv',
                        help='input training data file name')

    args = parser.parse_args()
    
    #%% Path
    fpath = "./data"
    fname = "data_set.csv"
    
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
        mean_x = np.mean(x, axis = 0)
        std_x = np.std(x, axis = 0)
        for i in range(len(x)):
            for j in range(len(x[0])):
                if std_x[j] != 0:
                    y[i][j] = (x[i][j] - mean_x[j]) / std_x[j]
        return y
    
    def _al_nor(x):
        y = np.copy(x)
        for i in range(x.shape[0]):
            y[i, :, :] = _nor(x[i, :, :])
        return y
            
    
    #%% Read csv
    Data = np.array(pd.read_csv(os.path.join(fpath, fname)))
    train_data = Data[:701, 4:]
    train_label = Data[60:761, 4]
    test_data = Data[701:761, 4:]
    val_label = Data[761:797, 4]
    
    tra_D_p, tra_L_p = _pack(train_data, train_label, H)
    train_data_nor, test_data_nor = _al_nor(tra_D_p), _nor(test_data)
    
    #%% Save
    np.save("train_data.npy", train_data_nor)
    np.save("train_label.npy", tra_L_p)
    np.save("val_label.npy", val_label)
    np.save("test_data.npy", test_data_nor)
