import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import matplotlib.font_manager as fm

forecastNum = 14
targetName = '備轉容量(MW)'

df = pd.read_csv('./data/electricity_data.csv')



# Create Training and Test
train = df[targetName][:-forecastNum]
test = df[targetName][-forecastNum-7:]

traininput = np.zeros([0, 7])
trainlabel = np.zeros([0, 7])
for n in range(0, train.count()-13, 7):
    traininput = np.concatenate(
        (traininput, train[n:n+7].values.reshape(1, -1)), axis=0)
    trainlabel = np.concatenate(
        (trainlabel, train[n+7:n+7+7].values.reshape(1, -1)), axis=0)
testinput = np.zeros([0, 7])
testlabel = np.zeros([0, 7])
for n in range(0, test.count()-13, 7):
    testinput = np.concatenate(
        (testinput, test[n:n+7].values.reshape(1, -1)), axis=0)
    testlabel = np.concatenate(
        (testlabel, test[n+7:n+7+7].values.reshape(1, -1)), axis=0)

print(testlabel.shape)
regr = MLPRegressor(random_state=1, max_iter=1000).fit(traininput, trainlabel)
pred = regr.predict(testinput)

# PLOT
fontPath = './data/NotoSansTC-Regular.otf'
font_legend = fm.FontProperties(fname=fontPath, size=16)
font_ticks = fm.FontProperties(fname=fontPath, size=11)
font_title = fm.FontProperties(fname=fontPath, size=16)

plt.plot(testlabel.flatten(), label='true')
plt.plot(pred.flatten(), label='pred')

rmse = np.round(np.sqrt(np.mean(np.power(pred.flatten()-testlabel.flatten(), 2))), 4)
plt.title(f'{targetName}\nRMSE = {rmse}', fontproperties=font_title)
plt.xticks(ticks=np.arange(forecastNum), labels=df['date'][-forecastNum:],
           fontproperties=font_ticks, rotation=60)
plt.legend(loc=1, prop=font_legend)
plt.show()
