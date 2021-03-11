import pandas as pd
from datetime import datetime


df = pd.read_csv('./data/台灣電力公司_過去電力供需資訊.csv')

df['weekcount'] = [datetime.strptime(str(a), "%Y%m%d").isocalendar()[1] for a in df['日期'].values] 
df['daycount'] = [datetime.strptime(str(a), "%Y%m%d").isocalendar()[2] for a in df['日期'].values]
df['year'] = [datetime.strptime(str(a), "%Y%m%d").year for a in df['日期'].values]

df = df[['year','weekcount','daycount','淨尖峰供電能力(MW)','尖峰負載(MW)','備轉容量(MW)']]
# df = df.rename(columns={'淨尖峰供電能力(MW)':'supply','尖峰負載(MW)':'load','備轉容量(MW)':'remain'})

import matplotlib.pyplot as plt

plt.title('備轉容量(MW)')
plt.plot(df[df['year']==2019]['備轉容量(MW)'].values,label='2019')
plt.plot(df[df['year']==2020]['備轉容量(MW)'].values,label='2020')
plt.legend()
plt.show()

exit()
df.to_csv('./data/training_data.csv')