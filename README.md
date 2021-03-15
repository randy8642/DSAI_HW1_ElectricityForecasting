# DSAI_HW1_ElectricityForecasting
NCKU DSAI course homework

## 說明
* 說明連結\
[Dropbox paper](https://www.dropbox.com/scl/fi/tx7md0teq0z4m3v20h5cp/DSAI-HW1-Electricity-Forecasting.paper?dl=0&rlkey=ajmzfqg0bjivr9bmcu8mqhv72)
* 目標\
預測 2021/3/23 - 2021/3/29 台灣的電力備轉容量

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

## 資料來源
* [台灣電力公司_近三年每日尖峰備轉容量率](https://data.gov.tw/dataset/24945)
    * 2014/01/01 - 2020/12/31
* [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)
    * 2019/01/01 - 2021/01/31
* [台灣電力公司_本年度每日尖峰備轉容量率](https://data.gov.tw/dataset/25850)
    * 2021/01/01 - 當前

## 前處理
* 透過下列公式反推資料
    * 備轉容量 = 系統運轉淨尖峰能力 - 系統瞬時尖峰負載
    * 備轉容量率 = (備轉容量 ÷ 系統瞬時尖峰負載) × 100%

![](/img/supply_load_remain.png)

## 模型架構
