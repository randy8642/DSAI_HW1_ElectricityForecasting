import numpy as np
import pandas as pd


df = pd.read_csv('./data/electricity_data.csv')

df = df[['Day of the week', 'work', '備轉容量(MW)','瞬時尖峰負載(MW)','淨尖峰供電能力(MW)']]
df['work'] = [1 if i else 0 for i in df['work']]
print(df)

exit()
train_x = np.zeros([0, 3])
train_y = np.zeros([0, 7])
a = df.to_numpy()
# n=0時為周二(2019/01/01)
for n in range(6, a.shape[0]-8, 7):
    train_x = np.concatenate((train_x, a[n, :].reshape(1, 3)), axis=0)
    train_y = np.concatenate(
        (train_y, a[n+1:n+1+7, -1].reshape(1, 7)), axis=0)

print(train_x[-1,0])
print(train_x.shape)
print(train_y.shape)

np.savez_compressed('./data/trainData', train_x=train_x, train_y=train_y)
