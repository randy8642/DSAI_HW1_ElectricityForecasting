import numpy as np
import pandas as pd


df = pd.read_csv('./data/每日備轉容量.csv')

df = df[['Day of the week', 'work', '備轉容量(MW)']]
df['work'] = [1 if i else 0 for i in df['work']]

# def filter(data):
#     fft = np.fft.fft(data)
#     fftf = np.fft.fftfreq(data.shape[0])
#     fft[abs(fftf) < 0.03] = 0
#     return np.fft.ifft(fft).real
# df['備轉容量(MW)'] = filter(df['備轉容量(MW)'])

train_x = np.zeros([0, 15])
train_y = np.zeros([0, 7])
a = df.to_numpy()
# n=6時為周二(2019/01/08)
for n in range(6, a.shape[0]-8, 7):
    train_x = np.concatenate((train_x,a[n]))


print(train_x.shape)
print(train_y.shape)

np.savez_compressed('./data/trainData', train_x=train_x, train_y=train_y)
