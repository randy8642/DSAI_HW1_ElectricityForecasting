import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from pandas.core.algorithms import value_counts

df = pd.read_csv('./data/假日表.csv')
df['weekend'] = ((df['Day of the week'] == '週六') | (df['Day of the week'] ==  '週日'))
df['weekday'] = [not i for i in df['weekend'].values]
a = []
for i in df['Day of the week'].values:
    if i == '週一':
        a.append(1)
    elif i == '週二':
        a.append(2)
    elif i == '週三':
        a.append(3)
    elif i == '週四':
        a.append(4)
    elif i == '週五':
        a.append(5)
    elif i == '週六':
        a.append(6)
    elif i == '週日':
        a.append(7)
print(len(a))
print(df['Day of the week'].shape)
df['Day of the week'] = a
df.to_csv('a.csv')
print(df)
exit()
df = pd.read_csv('./data/台灣電力公司_過去電力供需資訊.csv')

df['weekcount'] = [datetime.strptime(str(a), "%Y%m%d").isocalendar()[
    1] for a in df['日期'].values]
df['daycount'] = [datetime.strptime(str(a), "%Y%m%d").isocalendar()[
    2] for a in df['日期'].values]
df['year'] = [datetime.strptime(
    str(a), "%Y%m%d").year for a in df['日期'].values]

df = df[['year', 'weekcount', 'daycount',
         '淨尖峰供電能力(MW)', '尖峰負載(MW)', '備轉容量(MW)']]
df = df.rename(columns={'淨尖峰供電能力(MW)': 'supply',
                        '尖峰負載(MW)': 'load', '備轉容量(MW)': 'remain'})


a = df[['year', 'weekcount', 'daycount', 'remain']]
d = a.to_numpy()

np.save('./data/trainData.npy', d)
