import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

EPOCH = 3000
LEARNING_RATE = 1e-3


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
    
    


if __name__ == '__main__':
    main()
