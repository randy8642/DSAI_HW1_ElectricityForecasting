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


