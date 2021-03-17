import numpy as np
import matplotlib.pyplot as plt


def rmse(pred, true):
    return np.sqrt(np.mean(np.power(pred-true, 2)))


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
    import pandas as pd
    df = pd.read_csv('./data/electricity_data.csv')
    targetName = '備轉容量(MW)'
    df = df[['date',  targetName]]
    df = df.rename(columns={'date': 'ds', targetName: 'y'})
    from predictModel.Prophet import forecastByProphet
    pred = forecastByProphet(df, 7)

    print(pred)


if __name__ == '__main__':
    main()
