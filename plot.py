import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# READ
df_workday = pd.read_csv('./data/假日表.csv')
df_ele = pd.read_csv('./data/台灣電力公司_過去電力供需資訊.csv')

df = pd.concat([df_workday, df_ele], axis=1)


def filter(data):
    fft = np.fft.fft(data)
    fftf = np.fft.fftfreq(data.shape[0])
    # fft[abs(fftf) > 0.035] = 0
    # fft[abs(fftf) < 0.03] = 0
    return np.fft.ifft(fft).real


# d = filter(df['淨尖峰供電能力(MW)']) - filter(df['尖峰負載(MW)'])
d = df['尖峰負載(MW)'].values
d = d[5:705].reshape(7, -1)


plt.plot(d, label='', alpha=0.3)
plt.plot(np.mean(d, axis=1), label='mean', alpha=1,linewidth=3,color='red')

plt.xticks(ticks=np.arange(0, 7, 1), labels=np.arange(1, 8, 1))
plt.xlabel('weekday')
plt.ylabel('load(MW)')
plt.legend(loc='upper right')
plt.grid()
plt.show()
