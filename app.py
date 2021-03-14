import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

EPOCH = 2000
LEARNING_RATE = 1e-1


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
    d = np.load('./data/trainData.npz')
    train_x = torch.from_numpy(d['train_x'][:-7]).type(torch.FloatTensor)
    train_y = torch.from_numpy(d['train_y'][:-7]).type(torch.FloatTensor)


    model = nn.Sequential(
        nn.Linear(4, 1),
        nn.Flatten(0, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lossFunction = nn.MSELoss()

    model.train()
    for epoch in range(EPOCH):
        print(f'epoch {epoch+1}/{EPOCH}', end='\r')

        pred = model(train_x)

        optimizer.zero_grad()
        loss = lossFunction(pred, train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            print(f'loss {loss.item()}')
    print('')
    

    # TEST
    y = model(train_x).detach().numpy()
    plt.plot(train_x[:, 0], train_y)
    plt.plot(train_x[:, 0], y)
    plt.show()


if __name__ == '__main__':
    main()
