  
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
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
from model import model_01
from preProcess import _PreProcess

#%%
tStart = time.time()
print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Parameters
batch = 8
lr = 1e-4
the = 180
acc = 0
Epoch = 1000000

#%% Functions
def _RMSE(pred, val):
    mse = ((pred[:9] - val)**2).mean()
    rmse = np.sqrt(mse)
    return rmse

#%% Get Data
TRA_data, TRA_label, TES_data, L_min, L_max, val_label = _PreProcess(args.training)

#%% Dataset
train_data = torch.from_numpy(TRA_data).type(torch.FloatTensor)
train_label = torch.from_numpy(TRA_label).type(torch.FloatTensor)
test_data = torch.from_numpy(TES_data).type(torch.FloatTensor)

train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch, shuffle=True)


#%% Train
model = model_01(30, 772)
optim = optim.SGD(model.parameters(), lr=lr)
loss_f = nn.MSELoss()

model.to(device)
loss_f.to(device)

# Training
print('\n------Training------')
for epoch in range(Epoch):
    if acc<the and epoch!=0:
        break
    else:
        for n, (Data, Label) in enumerate(train_dataloader):
            model.train()
            optim.zero_grad()
            data = Data
            valid = Label
            valid = valid.to(device)
            data = data.to(device)
    
            pred = model(data)
            
            loss = loss_f(pred, valid)
            
            loss.backward()
            optim.step()
    
        #print('\n------Val.------')
        model.eval()
        data = test_data
        data = data.to(device)
        pred = model(data)
        PRED = (pred*(L_max) + L_min).cpu().data.numpy()
# =============================================================================
#         PRED = pred.cpu().data.numpy()        
# =============================================================================
        PRED = PRED.squeeze()
        acc = _RMSE(PRED, val_label)
        with torch.no_grad():
            print('epoch[{}], loss:{:.4f}, val_acc:{:.4f}'.format(epoch+1, loss.item(), acc))
    
#%% Test
Date = []
for i in range(7):
    Date.append(20210323+i)
Value = PRED[23:].squeeze()

diction = {"Date": Date,
           "Value": Value
           }
select_df = pd.DataFrame(diction)
select_df.to_csv(args.output,index=0,header=0)

'''
model = ElectricityForecastingModel()
model.train(df_training)
df_result = model.predict(n_step=7)
df_result.to_csv(args.output, index=0)
'''