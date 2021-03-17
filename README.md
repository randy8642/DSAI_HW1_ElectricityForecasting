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
    1. `sudo apt-get update`
    2. `sudo apt-get install build-essential`
    3. `pip install -r requirements.txt`
3. 執行\
`python app.py --training training_data.csv --output submission.csv`

## 資料來源
* [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)
    * 2019/01/01 - 2021/01/31
* [台灣電力公司_本年度每日尖峰備轉容量率](https://data.gov.tw/dataset/25850)
    * 2021/01/01 - 當前

## 前處理
* 透過下列公式反推資料
    * 備轉容量 = 系統運轉淨尖峰能力 - 系統瞬時尖峰負載
    * 備轉容量率 = (備轉容量 ÷ 系統瞬時尖峰負載) × 100%

![](/img/supply_load_remain.png)

## 模型 - Prophet

[Prophet官方網站](https://facebook.github.io/prophet/)

**安裝步驟(python 3.6.4)**
* Windows 10
    1. 安裝Microsoft C++ Build Tools \
        [下載連結](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/)
    2. `pip install pystan==2.17.1.0 fbprophet==0.6`

* Ubuntu 16.04.3 LTS
    1. `apt-get update`
    2. `apt-get install build-essential`
    3. `pip install pystan==2.17.1.0 fbprophet==0.6`

**資料分析**

將資料拆解
![prophet_analize_remain](/img/prophet_analize_remain.png)

**預測結果**
1. 預測 備轉容量(MW)
    * RMSE = 142.6549

![prophet_predict_remain](/img/prophet_predict_remain.png)

2. 預測尖峰負載(MW) 並轉換為備轉容量(MW)
    * 這邊假設備轉容量率固定為10%
    * RMSE = 187.1624
    
![prophet_predict_load2remain](/img/prophet_predict_load2remain.png)

## 模型 - MLP

**預測結果**
1. 預測 備轉容量(MW)
    * RMSE = 149.8179

![mlp_predict_remain](/img/mlp_predict_remain.png)