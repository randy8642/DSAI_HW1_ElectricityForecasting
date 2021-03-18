import numpy as np
from sklearn.neural_network import MLPRegressor


def forecastByMLP(trainData, predictNum=7):
    '''
    input DataFrame include ['ds', 'y']
    '''

    train = trainData['y']

    x = np.zeros([0, predictNum])
    y = np.zeros([0, predictNum])
    for n in range(0, train.count()-(predictNum*2-1), predictNum):
        x = np.concatenate(
            (x, train[n:n+predictNum].values.reshape(1, -1)), axis=0)
        y = np.concatenate(
            (y, train[n+predictNum:n+predictNum+predictNum].values.reshape(1, -1)), axis=0)

    test_x = train[-predictNum:].values.reshape(1, -1)

    regr = MLPRegressor(random_state=1, max_iter=1000).fit(x, y)
    pred = regr.predict(test_x)

    return pred.flatten()
