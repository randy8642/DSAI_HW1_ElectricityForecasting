'''
https://iter01.com/511487.html
'''
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

df = pd.read_csv('./data/electricity_data.csv')
df = df['備轉容量率(%)']

forecastNum = 7

# Create Training and Test
train = df[:-forecastNum]
test = df[-forecastNum:]

# Build Model
model = ARIMA(train, order=(3, 1, 1))  
fitted = model.fit(disp=0)  
print(fitted.summary())

# Plot residual errors
residuals = pd.DataFrame(fitted.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
fitted.plot_predict(dynamic=False)
plt.show()

# Forecast
fc, se, conf = fitted.forecast(forecastNum, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')

plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()