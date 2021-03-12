import numpy as np
import pandas as pd


df_workday = pd.read_csv('./data/假日表.csv')
df_ele = pd.read_csv('./data/台灣電力公司_過去電力供需資訊.csv')

df = pd.concat([df_workday, df_ele], axis=1)
df = df[['year', 'month', 'day', 'Day of the week', 'work', '備轉容量(MW)']]
df['work'] = [1 if i else 0 for i in df['work']]

import matplotlib.pyplot as plt

data = df['備轉容量(MW)'].values
fft = np.fft.fft(data)
fftf = np.fft.fftfreq(data.shape[0])
fft[abs(fftf) > 0.005] = 0
d = np.fft.ifft(fft)

plt.plot(np.fft.fftshift(np.fft.fftfreq(data.shape[0])),abs(np.fft.fftshift(np.fft.fft(data))))
plt.show()
plt.plot(d)
plt.plot(data)
plt.show()



exit()

train_x = np.zeros([0, 7, 6])
train_y = np.zeros([0, 1])
a = df.to_numpy()
for n in range(0, a.shape[0]-7, 1):
    train_x = np.concatenate((train_x, a[n:n+7, :].reshape(1, 7, 6)), axis=0)
    train_y = np.concatenate((train_y, a[n+7, 5].reshape(1, 1)), axis=0)
print(train_x.shape)
print(train_y.shape)
np.savez_compressed('./data/trainData', train_x=train_x,train_y=train_y)
