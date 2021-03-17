  
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
from sklearn.neural_network import MLPRegressor
import time
from preProcess_3 import _PreProcess

#%%
tStart = time.time()

#%% Functions
def _RMSE(pred, val):
    mse = ((pred.squeeze() - val.squeeze())**2).mean()
    rmse = np.sqrt(mse)
    return rmse

#%% Get Data
TRA_data, TRA_label, VAL_data, VAL_label, TES_data = _PreProcess(args.training)
leng = TRA_data.shape[0]

#%% Train
model = MLPRegressor(random_state=1, hidden_layer_sizes=(2), activation="relu",solver='adam', batch_size=16, learning_rate="constant",
                     learning_rate_init=1e-3, max_iter=500)
model.fit(TRA_data.reshape(leng, -1), TRA_label)

# Val
val_pred = model.predict(VAL_data.reshape(1, -1))
val_rmse = _RMSE(val_pred, VAL_label)
print("VAL_RMSE >>", val_rmse)

#%% Test
PRED = model.predict(TES_data.reshape(1, -1))
Date = []
for i in range(7):
    Date.append(20210323+i)
Value = PRED.squeeze()

diction = {"Date": Date,
           "Value": Value[7:]
           }
select_df = pd.DataFrame(diction)
select_df.to_csv(args.output,index=0,header=0)


