import pandas as pd

#
df = pd.read_csv('./data/electricity_data.csv')
targetName = '備轉容量(MW)'

#
df = df[['date',  targetName]]
df = df.rename(columns={'date': 'ds', targetName: 'y'})

#
df.to_csv('./trainingData.csv')