from model import ElectricityForecastingModel
import torch
import pandas as pd

EPOCH = 30
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
    df = pd.read_csv(args.training)
    train_set = torch.from_numpy(
        df[['year', 'month', 'day', 'supply', 'load']].values).type(torch.FloatTensor)
    trainLoader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)

    # model & opt & lossfunc
    model = ElectricityForecastingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.MSELoss()

    # TRAIN
    train(model=model, dataloader=trainLoader,
          optimizer=optimizer, lossFunction=loss_function)

    # PREDICT
    a = predict()


def train(model, dataloader, optimizer, lossFunction):
    for epoch in range(EPOCH):
        epoch += 1

        model.train()

        for n, data in enumerate(dataloader):
            year, month, day, supply, load = data[:,
                                                  0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]

            inputData = torch.cat(
                (year.reshape(-1, 1), month.reshape(-1, 1), day.reshape(-1, 1)), dim=1)
            targetData = torch.cat(
                (supply.reshape(-1, 1), load.reshape(-1, 1)), dim=1)

            pred = model(inputData)            

            # Backward
            optimizer.zero_grad()

            loss = lossFunction(pred, targetData) 

            loss.backward()

            optimizer.step()

        with torch.no_grad():
            print(loss.item())

    pass


def predict():

    pass


if __name__ == '__main__':
    main()
