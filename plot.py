import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# READ

df = pd.read_csv('./data/每日備轉容量.csv')


def filter(data):
    fft = np.fft.fft(data)
    fftf = np.fft.fftfreq(data.shape[0])
    # fft[abs(fftf) > 0.035] = 0
    fft[abs(fftf) < 0.14] = 0
    return np.fft.ifft(fft).real


d = df['備轉容量(MW)'].values
print( df.iloc[[6]])

d = d[6:783].reshape(7, -1)

# plt.plot(d)
[b, c, d, e, f, g, h] = plt.plot(d.T)
plt.legend([b, c, d, e, f, g, h], [ 'Tue.','Wen.', 'Thur.', 'Fri.', 'Sat.', 'Sun.', 'Mon.'], loc='upper right')
plt.suptitle('Remain(MW)')
plt.show()
