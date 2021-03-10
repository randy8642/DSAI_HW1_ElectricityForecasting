import pandas as pd

df = pd.read_csv('./data/台灣電力公司_過去電力供需資訊.csv')



df['年'] = [int(str(a)[:4]) for a in df['日期'].values]
df['月'] = [int(str(a)[4:6]) for a in df['日期'].values]
df['日'] = [int(str(a)[6:8]) for a in df['日期'].values]
df = df[['年','月','日','淨尖峰供電能力(MW)','尖峰負載(MW)','備轉容量(MW)']]
df = df.rename(columns={"年": "year", "月": "month","日": "day","淨尖峰供電能力(MW)": "supply","尖峰負載(MW)": "load","備轉容量(MW)": "remain"})

df.to_csv('./data/training_data.csv')