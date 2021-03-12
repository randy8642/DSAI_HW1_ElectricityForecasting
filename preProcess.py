import numpy as np
import pandas as pd


df_workday = pd.read_csv('./data/假日表.csv')
df_ele = pd.read_csv('./data/台灣電力公司_過去電力供需資訊.csv')

df = pd.concat([df_workday, df_ele], axis=1)
df = df[['year', 'month', 'day', 'Day of the week', 'work', '備轉容量(MW)']]
df['work'] = [1 if i else 0 for i in df['work']]

train = np.zeros([0, 7*6+1])
a = df.to_numpy()
for n in range(0, a.shape[0]-7, 1):
    input = a[n:n+7, :].flatten()
    label = a[n+7, 5:]
    t = np.concatenate((input, label), axis=0).reshape(1, -1)
    train = np.concatenate((train, t), axis=0)
print(train.shape)
np.save('./data/trainData.npy', train)
