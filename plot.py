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


d = filter(df['備轉容量(MW)'])
# d = (df['備轉容量(MW)']).values


plt.plot(d)
plt.show()

exit()
aaa = np.zeros([0,7])
for n in np.arange(0, 777, 7):
    aaa = np.concatenate((aaa, (d[n+7:n+7+7]-d[n:n+7].mean()).reshape(1,7)), axis=0)
print(aaa)


plt.plot(aaa.T)
plt.xticks(ticks=range(7), labels=[
           'Tue.', 'Wen.', 'Thur.', 'Fri.', 'Sat.', 'Sun.', 'Mon.'])
plt.suptitle('Remain(MW)')
plt.show()
