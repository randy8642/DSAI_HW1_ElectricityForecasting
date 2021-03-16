import pandas as pd
import numpy as np



df = pd.read_csv('./data/electricity_data.csv')

df = df[['date',  '瞬時尖峰負載(MW)']]
df = df.rename(columns={'date':'ds','瞬時尖峰負載(MW)':'y'})