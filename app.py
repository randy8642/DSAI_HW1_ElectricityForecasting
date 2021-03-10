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
import time
from model import model_01
from preProcess import _PreProcess

#%%
tStart = time.time()
print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Parameters
batch = 8
lr = 1e-20
Epoch = 30

#%% Get Data
train_data_nor, tra_L_p, val_label, test_data_nor = _PreProcess(args.training)

#%% Dataset
train_data = torch.from_numpy(train_data_nor).type(torch.FloatTensor)
train_label = torch.from_numpy(tra_L_p).type(torch.FloatTensor)
test_data = torch.from_numpy(test_data_nor).type(torch.FloatTensor)

train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch, shuffle=True)

#%% Train
model = model_01(4, 20, 3)
optim = optim.Adam(model.parameters(), lr=lr)
loss_f = nn.MSELoss()

model.to(device)
loss_f.to(device)

# Training
print('\n------Training------')
model.train()
for epoch in range(Epoch):
    for n, (Data, Label) in enumerate(train_dataloader):
        optim.zero_grad()
        data = Data
        valid = Label
        valid = valid.to(device)
        data = data.to(device)

        pred = model(data)
        
        loss = loss_f(pred, valid)
        
        loss.backward()
        optim.step()
        
    with torch.no_grad():
        print('epoch[{}], loss:{:.4f}'.format(epoch+1, loss.item()))

'''
model = ElectricityForecastingModel()
model.train(df_training)
df_result = model.predict(n_step=7)
df_result.to_csv(args.output, index=0)

'''