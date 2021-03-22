
# You should not modify this part, but additional arguments are allowed.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-T','--training',
                   default='training_data.csv',
                   help='input training data file name')

parser.add_argument('--output',
                    default='submission.csv',
                    help='output file name')

parser.add_argument('-M', '--model',
                    default='prophet',
                    help='type of model')


args = parser.parse_args()

#%%
# The following part is an example.
# You can modify it at will.
import numpy as np
import pandas as pd
import time
from PreProcess import _PreProcess, _PreProcess2

#%%
tStart = time.time()

#%% Parameters
batch = 16
lr = 1e-3
the = 120
val_rmse = 180
Epoch = 1000

#%% Functions
def _RMSE(pred, val):
    mse = ((pred - val)**2).mean()
    rmse = np.sqrt(mse)
    return rmse

#%% Get Data
TRA_data, TRA_label, VAL_data, val_label, TES_data = _PreProcess(args.training)
VAL_data2, TES_data2 = _PreProcess2("training_data2.csv")
leng = TRA_data.shape[0]

#%% Torch
if args.model == 'pytorch':
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import random
    from model import m02


    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(20)

    train_data = torch.from_numpy(TRA_data).type(torch.FloatTensor)
    train_label = torch.from_numpy(TRA_label).type(torch.FloatTensor)
    val_data = torch.from_numpy(VAL_data).type(torch.FloatTensor)
    test_data = torch.from_numpy(TES_data).type(torch.FloatTensor)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size=batch, shuffle=True)

    # Device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Device >> ", device)

    # Train
    model = m02(3, 30)
    optim = optim.Adam(model.parameters(), lr=lr)
    loss_F = nn.MSELoss()

    # model.to(device)
    # loss_F.to(device)

    print('\n------Training------')
    for epoch in range(Epoch):
        if epoch>500 and val_rmse<the:
            break
        else:
            model.train()
            for n, (Data, Label) in enumerate(train_dataloader):
                optim.zero_grad()

                # Data = Data.to(device)
                # Label = Label.to(device)
                Pred = model(Data)
                loss = loss_F(Pred, Label)

                loss.backward()
                optim.step()

            model.eval()
            with torch.no_grad():
                #  val_data = val_data.to(device)

                 val_pred = model(val_data)
                 val_pred = val_pred.cpu().data.numpy()

                 val_rmse = _RMSE(val_pred, val_label)

    print('epoch[{}], VAL_RMSE >>{:.4f}'.format(epoch+1, val_rmse))

    # Test
    print('\n------Testing------')
    model.eval()
    with torch.no_grad():
        #  test_data = test_data.to(device)
         PRED = model(test_data)
         PRED = PRED.cpu().data.numpy()

#%% Sklearn
elif args.model == 'sklearn':
    from sklearn.neural_network import MLPRegressor
    # Train
    print('\n------Training------')
    model = MLPRegressor(random_state=1, hidden_layer_sizes=(2), activation="relu",solver='adam', batch_size=batch, learning_rate="constant",
                         learning_rate_init=lr, max_iter=Epoch)
    model.fit(TRA_data.reshape(leng, -1), TRA_label)

    # Val
    val_pred = model.predict(VAL_data.reshape(1, -1))
    val_rmse = _RMSE(val_pred, val_label)
    print("VAL_RMSE >>", round(val_rmse, 4))

    # Test
    print('\n------Testing------')
    PRED = model.predict(TES_data.reshape(1, -1))


#%% Prophet
elif args.model == 'prophet':
    from prophet import forecastByProphet
    # Train
    print('\n------Training------')
    val_pred = forecastByProphet(VAL_data2, 8)
    val_rmse = _RMSE(val_pred, val_label)
    print("VAL_RMSE >>", round(val_rmse, 4))

    # Test
    print('\n------Testing------')
    PRED = forecastByProphet(TES_data2, 8)

#%% Else
else:
    print("Could not use this model. See model list on README.")

#%% Save
np.save(args.model + "_" + "val_pred.npy", val_pred)

Date = []
for i in range(7):
    Date.append(20210323+i)
Value = PRED.squeeze()

print("===TEST_PRED===")
print(Value)

diction = {"Date": Date,
           "Value": Value[1:]
           }
select_df = pd.DataFrame(diction)
# sf = args.model + "_" + args.output
sf = args.output
select_df.to_csv(sf,index=0,header=0)

#%%
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))
