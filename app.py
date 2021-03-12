import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import sklearn.metrics as metrics

EPOCH = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4


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
    train_x = torch.from_numpy(d['train_x'][:700]).type(torch.FloatTensor)
    train_y = torch.from_numpy(d['train_y'][:700]).type(torch.FloatTensor)
    trainDataSet = data.TensorDataset(train_x, train_y)
    trainLoader = data.DataLoader(
        trainDataSet, batch_size=BATCH_SIZE, shuffle=True)

    model = m()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    lossFunction = nn.MSELoss()

    model.train()
    for epoch in range(EPOCH):
        print(f'epoch {epoch+1}/{EPOCH}', end='\r')
        for n, (x, y) in enumerate(trainLoader):

            pred = model(x)

            optimizer.zero_grad()
            loss = lossFunction(pred, y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            print(f'loss {loss.item()}')
    print('')

    # TEST
    test_x = torch.from_numpy(d['train_x'][700:]).type(torch.FloatTensor)
    test_y = torch.from_numpy(d['train_y'][700:]).type(torch.FloatTensor)
    testDataSet = data.TensorDataset(test_x, test_y)
    testLoader = data.DataLoader(
        testDataSet, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        model.eval()
        pred = torch.zeros([0, 1])
        for n, (x, y) in enumerate(testLoader):
            out = model(x)
            pred = torch.cat((out, pred), dim=0)

       
        r2 = metrics.r2_score(y_true=test_y.numpy(), y_pred=pred.numpy())
        print(test_y.numpy().flatten())
        print(pred.numpy().flatten())
        print(r2)


class m(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(42, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    main()
