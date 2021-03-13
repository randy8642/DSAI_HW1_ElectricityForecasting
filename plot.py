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
    fft[abs(fftf) < 0.03] = 0
    return np.fft.ifft(fft).real


# PLOT
# plt.plot(filter(df['尖峰負載(MW)']), label='load(MW)')
# plt.plot(filter(df['淨尖峰供電能力(MW)']), label='supply(MW)')
plt.plot(filter(df['淨尖峰供電能力(MW)']) -
         filter(df['尖峰負載(MW)']), label='remain(MW)')
plt.xticks(ticks=range(0, df['date'].shape[0], 30),
           labels=df['date'][::30], rotation=45)
plt.suptitle('filtered with 365days')
plt.legend(loc='upper right')
plt.grid()
plt.show()
