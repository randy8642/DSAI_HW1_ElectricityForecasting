'''
https://facebook.github.io/prophet/docs/quick_start.html#python-api
'''
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fbprophet import Prophet

forecastNum = 14
targetName = '備轉容量(MW)'

df = pd.read_csv('./data/electricity_data.csv')

df = df[['date',  targetName]]
df = df.rename(columns={'date': 'ds', targetName: 'y'})

# Create Training and Test
train = df[:-forecastNum]
test = df[-forecastNum:]

m = Prophet()
m.fit(train)

future = m.make_future_dataframe(periods=forecastNum)


forecast = m.predict(future)
forecast.to_csv('prephet_predict.csv')

# PLOT
df_forecast = forecast[['yhat']]
fontPath = './data/NotoSansTC-Regular.otf'
font_legend = fm.FontProperties(fname=fontPath, size=16)
font_ticks = fm.FontProperties(fname=fontPath, size=11)
font_title = fm.FontProperties(fname=fontPath, size=16)

pred = df_forecast[-forecastNum:].values.flatten()
true = test['y'].values.flatten()
plt.plot(pred, label='forecast')
plt.plot(true, label='true')
rmse = np.round(np.sqrt(np.mean(np.power(pred-true, 2))), 4)
plt.title(f'{targetName}\nRMSE = {rmse}', fontproperties=font_title)
plt.xticks(ticks=np.arange(forecastNum), labels=test['ds'],
           fontproperties=font_ticks, rotation=60)
plt.legend(loc=1, prop=font_legend)
plt.show()

fig, ax = plt.subplots(5, 1, sharex=True)
ax[0].plot(df['y'])
ax[0].set_ylabel('raw',fontproperties=font_legend)
ax[1].plot(forecast['trend'])
ax[1].set_ylabel('trend',fontproperties=font_legend)
ax[2].plot(forecast['weekly'])
ax[2].set_ylabel('weekly',fontproperties=font_legend)
ax[3].plot(forecast['yearly'])
ax[3].set_ylabel('yearly',fontproperties=font_legend)
ax[4].plot(df['y'] - forecast['trend'] -
           forecast['weekly'] - forecast['yearly'])
ax[4].set_ylabel('residual',fontproperties=font_legend)
fig.suptitle(f'{targetName}',fontproperties=font_title)
plt.show()
