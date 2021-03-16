import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
fontPath = './data/NotoSansTC-Regular.otf'
font_legend = fm.FontProperties(fname=fontPath, size=16)
font_ticks = fm.FontProperties(fname=fontPath, size=11)
font_title = fm.FontProperties(fname=fontPath, size=20)


df = pd.read_csv('./data/electricity_data.csv')

# df = df[['Day of the week', 'work', '備轉容量(MW)','瞬時尖峰負載(MW)','淨尖峰供電能力(MW)']]
# df['work'] = [1 if i else 0 for i in df['work']]


def splitWeek(data):
    week = np.zeros([0, 7])
    for i in np.arange(0, data.shape[0]-7, 7):
        week = np.concatenate((week, data[i:i+7].reshape(1, -1)), axis=0)
    return week


d = splitWeek(df['淨尖峰供電能力(MW)'].values)

# 移除平均
d = d - np.repeat(np.mean(d, axis=1, keepdims=True), 7, axis=1)

# d = df['備轉容量率(%)']

plt.plot(d.T, alpha=0.3)

plt.xticks(ticks=np.arange(7), labels=['Mon.', 'Tue.', 'Wen.', 'Thur.', 'Fri.', 'Sat.', 'Sun.'], fontproperties=font_ticks)


# plt.xticks(ticks=np.arange(804)[::20], labels=df['date'].values[::20], rotation=60, fontproperties=font_ticks)
plt.title('淨尖峰供電能力(MW)[當週] - 平均[當週]',fontproperties=font_title)
plt.grid()
# plt.legend(loc=1, prop=font_legend)
plt.show()






