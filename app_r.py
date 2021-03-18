import numpy as np
import matplotlib.pyplot as plt


def rmse(pred, true):
    return np.sqrt(np.mean(np.power(pred-true, 2)))


def main():
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--training',
                        default='trainingData.csv',
                        help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')

    parser.add_argument('--model',
                        default='mlp',
                        help='')

    parser.add_argument('--predict_num',
                        default=14,
                        help='')

    args = parser.parse_args()

    #　TRAIN
    import pandas as pd
    from datetime import datetime, timedelta
    df = pd.read_csv(args.training)
    num = args.predict_num

    # GET DATE
    trainLastDate = df['ds'].tail(1).values[0]
    trainLastDate = datetime.strptime(trainLastDate, "%Y/%m/%d")

    predDate = []
    for n in np.arange(num):
        n += 1
        predDate.append(
            (trainLastDate + timedelta(days=n)).strftime('%Y/%m/%d'))

    # GET PREDICT
    if args.model == 'prophet':
        from predictModel.Prophet import forecastByProphet
        pred = forecastByProphet(df, num)
    elif args.model == 'mlp':
        from predictModel.MLP import forecastByMLP
        pred = forecastByMLP(df, num)

    
    # OUTPUT
    pred = np.round(pred, 0)
    df_pred = pd.DataFrame({'date': predDate, 'predict': pred})
    
    df_pred.to_csv(args.output)


if __name__ == '__main__':
    main()
