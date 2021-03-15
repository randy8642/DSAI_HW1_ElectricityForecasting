import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

EPOCH = 3000
LEARNING_RATE = 1e-3

def rmse(pred,true):
    return np.sqrt(np.mean(np.power(pred-true,2)))

def main():
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training',
                        default='./data/training_data.csv',
                        help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    #　TRAIN
    d = np.load('./data/trainData.npz', allow_pickle=True)
    trainTestCut = 4
    train_x = torch.from_numpy(d['train_x'][:-trainTestCut, :]).type(torch.FloatTensor)
    train_y = torch.from_numpy(d['train_y'][:-trainTestCut, :]).type(torch.FloatTensor)
    test_x = torch.from_numpy(d['train_x'][-trainTestCut:, :]).type(torch.FloatTensor)
    test_y = torch.from_numpy(d['train_y'][-trainTestCut:, :]).type(torch.FloatTensor)

    model = nn.Sequential(
        nn.Linear(14, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(20, 7),
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lossFunction = nn.MSELoss()

    model.train()
    for epoch in range(EPOCH):
        print(f'epoch {epoch+1}/{EPOCH}', end='\r')

        pred = model(train_x[:,:14])

        optimizer.zero_grad()
        loss = lossFunction(pred, train_y)
        loss.backward()
        optimizer.step()
        # with torch.no_grad():
        #     print(f'loss {loss.item()}')
    print('')

    # TEST
    scale = test_x[:,-1].numpy().reshape(trainTestCut,1)

    pred_y = model(test_x[:,:14]).detach().numpy() * scale
    test_y = test_y.numpy() * scale

    pred_y = pred_y.flatten()
    test_y = test_y.flatten()
    

    
    print(rmse(test_y, pred_y))
    
    plt.plot(pred_y, label='pred')
    plt.plot(test_y, label='true')
    plt.xticks(ticks=range(7*trainTestCut), labels=[
               'Wen.', 'Thur.', 'Fri.', 'Sat.', 'Sun.', 'Mon.', 'Tue.']*trainTestCut)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
