'''
https://facebook.github.io/prophet/docs/quick_start.html#python-api
'''
import pandas as pd
import numpy as np
from fbprophet import Prophet


df = pd.read_csv('./data/electricity_data.csv')

df = df[['date',  '瞬時尖峰負載(MW)']]
df = df.rename(columns={'date':'ds','瞬時尖峰負載(MW)':'y'})



m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=30)


forecast = m.predict(future)
forecast.to_csv('pridict.csv')
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
# fig1 = m.plot(forecast)

# fig2 = m.plot_components(forecast)
