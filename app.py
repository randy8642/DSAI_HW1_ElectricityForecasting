from model import ElectricityForecastingModel
import torch
import pandas as pd
import numpy as np

EPOCH = 200
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

    # prepare dataset
    # df = pd.read_csv(args.training)
    data = np.load('./data/trainData.npy')
    train_set = torch.from_numpy(data[:700]).type(torch.FloatTensor)
    trainLoader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_set = torch.from_numpy(data[700:]).type(torch.FloatTensor)
    testLoader = torch.utils.data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False)

    # model & opt & lossfunc
    model = ElectricityForecastingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.MSELoss()

    # TRAIN
    train(model=model, dataloader=trainLoader,
          optimizer=optimizer, lossFunction=loss_function)

    # PREDICT
    a = predict(model=model, dataloader=testLoader)


def train(model, dataloader, optimizer, lossFunction):
    for epoch in range(EPOCH):
        epoch += 1

        model.train()

        for n, data in enumerate(dataloader):
            

            

            pred = model()

            # Backward
            optimizer.zero_grad()

            loss = lossFunction(pred, (supply - load)/10000)

            loss.backward()

            optimizer.step()

        with torch.no_grad():
            print(loss.item())

    pass


def predict(model, dataloader):
    model.eval()

    pred = np.zeros([0, 1])
    target = np.zeros([0, 1])
    for n, data in enumerate(dataloader):
        year, weekcount, daycount, supply, load = data[:,
                                                       0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

        weekcount = weekcount.type(torch.LongTensor)
        daycount = daycount.type(torch.LongTensor)

        out = model(year, weekcount, daycount)

        out = out.detach().numpy().reshape(-1, 1)

        pred = np.concatenate((pred, out), axis=0)
        target = np.concatenate(
            (target, ((supply - load)/10000).reshape(-1, 1)), axis=0)

    print(pred.shape)
    print(target.shape)

    import matplotlib.pyplot as plt
    plt.plot(pred)
    plt.plot(target)
    plt.show()


if __name__ == '__main__':
    main()
