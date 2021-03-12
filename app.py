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
    from sklearn.neural_network import MLPRegressor
    d = np.load('./data/trainData.npy')
    n = 20
    x = d[:-n, :-1]
    y = d[:-n, -1:]
    reg = MLPRegressor(random_state=1, max_iter=1000)
    reg.fit(x, y)
    # r2
    print(reg.score(x, y))

    # predict
    test_x = d[-n:, :-1]
    test_y = d[-n:, -1:].flatten()
    pred = reg.predict(test_x)

    print(test_y)
    print(pred)
    from sklearn.metrics import r2_score
    print(r2_score(test_y, pred))


if __name__ == '__main__':
    main()
