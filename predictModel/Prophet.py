'''
https://facebook.github.io/prophet/docs/quick_start.html#python-api
'''
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fbprophet import Prophet


df = pd.read_csv('./data/electricity_data.csv')

df = df[['date',  '備轉容量(MW)']]
df = df.rename(columns={'date': 'ds', '備轉容量(MW)': 'y'})
forecastNum = 14
# Create Training and Test
train = df[:-forecastNum]
test = df[-forecastNum:]

m = Prophet()
m.fit(train)

future = m.make_future_dataframe(periods=forecastNum)


forecast = m.predict(future)
# forecast.to_csv('pridict.csv')
df_forecast = forecast[['yhat']]

# PLOT
fontPath = './data/NotoSansTC-Regular.otf'
font_legend = fm.FontProperties(fname=fontPath, size=16)
font_ticks = fm.FontProperties(fname=fontPath, size=11)
font_title = fm.FontProperties(fname=fontPath, size=16)

pred = df_forecast[-forecastNum:].values.flatten()
true = test['y'].values.flatten()
plt.plot(pred, label='forecast')
plt.plot(true, label='true')
rmse = np.round(np.sqrt(np.mean(np.power(pred-true, 2))),4)
plt.title(f'備轉容量(MW)\nRMSE = {rmse}', fontproperties=font_title)
plt.xticks(ticks=np.arange(forecastNum), labels=test['ds'],
           fontproperties=font_ticks, rotation=60)
plt.legend(loc=1, prop=font_legend)
plt.show()
