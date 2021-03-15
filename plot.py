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


# d = filter(df['備轉容量(MW)'])
d = (df['備轉容量(MW)']).values
plt.plot(d)
plt.show()

exit()
aaa = np.zeros([7, 0])
for n in range(0, 784, 7):
    aaa = np.concatenate((aaa, d[n:n+7].reshape(7, 1)), axis=1)
# d = d[6:783].reshape(7, -1)

plt.plot(aaa)
plt.xticks(ticks=range(7), labels=[
           'Tue.', 'Wen.', 'Thur.', 'Fri.', 'Sat.', 'Sun.', 'Mon.'])
plt.suptitle('Remain(MW)')
plt.show()
