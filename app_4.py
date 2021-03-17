  
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
import torch
import torch.optim as optim
import torch.nn as nn
import time
from preProcess_3 import _PreProcess
from model import m01, m02, m03

#%%
tStart = time.time()
print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#%% Parameters
batch = 16
lr = 1e-3
the = 150
val_rmse = 180
Epoch = 1000000

#%% Functions
def _RMSE(pred, val):
    mse = ((pred.squeeze() - val.squeeze())**2).mean()
    rmse = np.sqrt(mse)
    return rmse

#%% Get Data
TRA_data, TRA_label, VAL_data, VAL_label, TES_data = _PreProcess(args.training)
train_data = torch.from_numpy(TRA_data).type(torch.FloatTensor)
train_label = torch.from_numpy(TRA_label).type(torch.FloatTensor)
val_data = torch.from_numpy(VAL_data).type(torch.FloatTensor)
val_label = VAL_label
test_data = torch.from_numpy(TES_data).type(torch.FloatTensor)

train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch, shuffle=True)



#%% Train
#model = m01(3, 10, 30)
model = m02(3, 30)
optim = optim.Adam(model.parameters(), lr=lr)
loss_F = nn.MSELoss()

model.to(device)
loss_F.to(device)

print('\n------Training------')
for epoch in range(Epoch):
    if epoch>500 and val_rmse<the:
        break
    else:
        model.train()
        for n, (Data, Label) in enumerate(train_dataloader):
            optim.zero_grad()
            
            Data = Data.to(device)
            Label = Label.to(device)
            Pred = model(Data)
            loss = loss_F(Pred, Label)
            
            loss.backward()
            optim.step()
        
        model.eval()
        with torch.no_grad():
             val_data = val_data.to(device)
             
             val_pred = model(val_data)
             val_pred = val_pred.cpu().data.numpy()
             
             val_rmse = _RMSE(val_pred, val_label)
             print('epoch[{}], loss >>{:.4f}, RMSE >>{:.4f}'.format(epoch+1, loss.item(), val_rmse))
   
#%% Test
print('\n------Testing------')        
model.eval()
with torch.no_grad():
     test_data = test_data.to(device)
     PRED = model(test_data)
     PRED = PRED.cpu().data.numpy()

Date = []
for i in range(7):
    Date.append(20210323+i)
Value = PRED.squeeze()

diction = {"Date": Date,
           "Value": Value[7:]
           }
select_df = pd.DataFrame(diction)
select_df.to_csv(args.output,index=0,header=0)


