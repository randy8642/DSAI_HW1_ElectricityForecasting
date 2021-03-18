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
`python app.py --training training_data.csv --output submission.csv --model [model name]`
      * model種類：
         * pytoch (default)
         * sklearn

## 資料來源

### 主要資料

* [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)
    * 2019/01/02 - 2021/01/31
* [台灣電力公司_本年度每日尖峰備轉容量率](https://data.gov.tw/dataset/25850)
    * 2021/01/01 - 當前

### 其他資料

* [空氣品質監測日平均值(一般污染物)](https://data.epa.gov.tw/dataset/aqx_p_19)
    * 2019/01/01 - 當前 (缺少 2020/6/29 資料，故整理完後未使用)

## 分析

* 首先透過下列公式利用**備轉容量**以及**備轉容量率**反推**淨尖峰供電能力**與**尖峰負載**
    * 備轉容量 = 系統運轉淨尖峰能力 - 系統瞬時尖峰負載
    * 備轉容量率 = (備轉容量 ÷ 系統瞬時尖峰負載) × 100%

* 若將**淨尖峰供電能力**、**尖峰負載**以及**備轉容量率**依日期畫出趨勢圖，會呈以下現象：
    ![](https://i.imgur.com/XdUcAsU.png)
    另外將**備轉容量率**、**淨尖峰供電能力**與**尖峰負載**以一週為單位作圖，如下所示：
    ![](https://i.imgur.com/L08hMCN.png)
    ![](https://i.imgur.com/65CXQb7.png)
    ![](https://i.imgur.com/xlWnJfK.png)
    可以發現除**備轉容量率**之外，其餘兩者在每週的變化趨勢皆有相似的特徵；\
    若將**備轉容量**依星期順序拆成七條不同的折線圖，可得：
    ![](https://i.imgur.com/86KUEal.png)
>綜合上述兩點，我們認為**備轉容量**在每週的趨勢變化不大；\
>且與**備轉容量率**、**淨尖峰供電能力**與**尖峰負載**具有一定相關性，\
>因此我們打算採取以**前幾天之備轉容量率、淨尖峰供電能力和尖峰負載**，\
>預測**後一週或兩週的備轉容量**。

## 前處理

前處理方式可依模型分成下列兩種：
* [PreProcess.py](LINK)
    >training data的範圍從 2019/01/01 至 2021/02/15\
    >validation data的範圍從 2021/01/31 至 2021/03/01\
    >testing dara的範圍從 2021/02/14 至 2021/03/15

    上述資料依30天的間隔，以步進為一天的方式進行切割，\
    目的是利用前30天的資料，預測後14天的資料。

![](/img/supply_load_remain.png)

## 模型

* train
    2019/01/01 - 2021/02/28
* test
    2021/03/01 - 2021/03/14
### MLP
* 待補

### ARIMA
* 待補

### Prophet
[官方網站](https://facebook.github.io/prophet/)\
\
安裝步驟(python 3.6.4)
* Windows 10
    1. 安裝Microsoft C++ Build Tools \
        [下載連結](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/)
    2. `pip install pystan==2.17.1.0`
    3. `pip install fbprophet==0.6`

* Ubuntu 16.04.3 LTS
    1. 待補

預測結果
1. 直接預測 備轉容量(MW)
    * RMSE = 142.0756

![prophet_predict_remain](/img/prophet_predict_remain.png)

2. 預測尖峰負載(MW) 並轉換為備轉容量(MW)
    * 這邊假設備轉容量率固定為10%
    * RMSE = 187.1624

![prophet_predict_load2remain](/img/prophet_predict_load2remain.png)
