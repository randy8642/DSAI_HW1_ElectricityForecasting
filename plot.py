import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# READ

df = pd.read_csv('./data/每日備轉容量.csv')


def filter(data):
    fft = np.fft.fft(data)
    fftf = np.fft.fftfreq(data.shape[0])
    # fft[abs(fftf) > 0.035] = 0
    fft[abs(fftf) < 0.03] = 0
    return np.fft.ifft(fft).real


d = df['備轉容量(MW)'].values


d = d[6:783].reshape(7, -1)


[b, c, d, e, f, g, h] = plt.plot(d.T)
plt.legend([b, c, d, e, f, g, h], ['Wen.', 'Thur.', 'Fri.',
                                   'Sat.', 'Sun.', 'Mon.', 'Tue.'], loc='upper right')
plt.suptitle('Remain(MW)')
plt.show()
