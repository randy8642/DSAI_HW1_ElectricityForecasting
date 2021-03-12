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


    #　TRAIN
    import numpy as np
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = np.load('./data/trainData.npy')
    x = d[:-5,:3]
    y = d[:-5,-1:]
    reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))
    reg.fit(x, y)
    # r2
    print(reg.score(x, y))
    
    # predict
    print(np.round(reg.predict(d[-5:,:3])))
    print(d[-5:,-1:].flatten())
    


if __name__ == '__main__':
    main()
