# DSAI_HW1_ElectricityForecasting
NCKU DSAI course homework

## 說明
* 說明連結\
[Dropbox paper](https://www.dropbox.com/scl/fi/tx7md0teq0z4m3v20h5cp/DSAI-HW1-Electricity-Forecasting.paper?dl=0&rlkey=ajmzfqg0bjivr9bmcu8mqhv72)
* 目標\
預測 2021/3/24 - 2021/3/30 台灣的電力備轉容量

## 環境
* python 3.6.4
* Ubuntu 16.04.3 LTS

## 使用方式
1. 進入專案資料夾\
`cd /d [path/to/this/project]`
2. 安裝套件\
`pip install -r requirements.txt`
3. 執行\
`python app.py --training training_data.csv --output submission.csv`

## 分析
### 圖表
* 一週中變化
    * 淨尖峰供電能力(MW)\
    ![supply_in_week](/img/supply_in_week.png)
    * 尖峰負載(MW)\
    ![load_in_week](/img/load_in_week.png)
    * 備轉容量(MW)\
    ![remain_in_week](/img/remain_in_week.png)


## 資料前處理
將資料分為每周/每年學習
## 模型架構
