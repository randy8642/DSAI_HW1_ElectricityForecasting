  
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
from preProcess_2 import _PreProcess

#%%
tStart = time.time()


#%% Parameters
batch = 16
lr = 100
the = 180
acc = 0
Epoch = 1000000

#%% Functions
def _RMSE(pred, val):
    mse = ((pred[:9] - val)**2).mean()
    rmse = np.sqrt(mse)
    return rmse

#%% Get Data
TRA_data, TRA_label, VAL_data, VAL_label, TES_data, mu, std = _PreProcess(args.training)

#%% Train
model = LinearRegression(fit_intercept=True)
model.fit(TRA_data.reshape(740, -1), TRA_label)
w = model.coef_
b = model.intercept_

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


