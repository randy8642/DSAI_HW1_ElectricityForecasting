  
# You should not modify this part, but additional arguments are allowed.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-T','--training',
                   default='training_data.csv',
                   help='input training data file name')

parser.add_argument('--output',
                    default='submission.csv',
                    help='output file name')
args = parser.parse_args()

#%%
# The following part is an example.
# You can modify it at will.
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import time
from preProcess_3 import _PreProcess

#%%
tStart = time.time()

#%% Functions
def _RMSE(pred, val):
    mse = ((pred - val)**2).mean()
    rmse = np.sqrt(mse)
    return rmse

#%% Get Data
TRA_data, TRA_label, VAL_data, VAL_label, TES_data = _PreProcess(args.training)

#%% Train
# model
model = LinearRegression(fit_intercept=True)
W = []
B = []
# Train
for i in range(7):
    data = TRA_data[str(i)]
    label = TRA_label[str(i)]
    bz = data.shape[0]
    data = data.reshape(bz, -1)
    
    model.fit(data, label)
    W.append(model.coef_)
    B.append(model.intercept_)
    
# Val.
P = []
L = []
for j in range(7):
    data = VAL_data[str(j)].reshape(-1)
    pred = np.dot(data, W[j]) + B[j]
    P.append(pred)
    L.append(VAL_label[str(j)])
    
VAL_RMSE = _RMSE(np.array(P), np.array(L))
print("VAL_RMSE >>", VAL_RMSE)
    
'''
# Val
val_pred = np.dot(VAL_data.reshape(7, -1), w) + b
VAL_RMSE = _RMSE(val_pred, VAL_label)
print("VAL_RMSE >>", VAL_RMSE)

#%% Test
PRED = np.dot(TES_data.reshape(7, -1), w) + b

Date = []
for i in range(7):
    Date.append(20210323+i)
Value = PRED.squeeze()

diction = {"Date": Date,
           "Value": Value
           }
select_df = pd.DataFrame(diction)
select_df.to_csv(args.output,index=0,header=0)
'''

