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
         * sklearn
         * pytoch
         * prophet (default)

## 資料來源

### 主要資料

* [台灣電力公司_過去電力供需資訊](https://data.gov.tw/dataset/19995)
  * 2019/01/01 - 2021/01/31
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

    ![PPMW_PMW_%](https://i.imgur.com/XdUcAsU.png)

    另外將**備轉容量率**、**淨尖峰供電能力**與**尖峰負載**以一週為單位作圖，如下所示：

    ![%](https://i.imgur.com/L08hMCN.png)

    ![PPMW](https://i.imgur.com/65CXQb7.png)

    ![PMW](https://i.imgur.com/xlWnJfK.png)

    可以發現除**備轉容量率**之外，其餘兩者在每週的變化趨勢皆有相似的特徵。

* 此外另外將**備轉容量**依星期順序拆成七條不同的折線圖，可得：

    ![MW](https://i.imgur.com/86KUEal.png)

>綜合上述三點，我們認為**備轉容量**在每週的趨勢變化不大；\
>且與**備轉容量率**、**淨尖峰供電能力**與**尖峰負載**具有一定相關性，\
>因此我們打算採取以**前幾天之備轉容量率、淨尖峰供電能力和尖峰負載**，\
>預測**後一週或兩週的備轉容量**。

## 前處理

前處理方式可依模型分成下列兩種：

* [PreProcess.__PreProcess](LINK)
    >training data的範圍從 2019/01/02 至 2021/01/20\
    >validation data的範圍從 2021/01/21 至 2021/02/19\
    >testing dara的範圍從 2021/02/20 至 2021/03/21

    上述資料依30天的間隔，以步進為一天的方式進行切割，\
    目的是利用前30天的資料，預測後8天的資料。\
    故各資料的維度如下：
    |            | Training   | Validation | Testing  |
    |------------|------------|------------|----------|
    | Data Dim.  | (721,30,3) | (1,30,3)   | (1,30,3) |
    | Label Dim. | (721,8)    | (8)        |          |

* [PreProcess.__PreProcess2](LINK)
    >validation data的範圍從 2019/01/02 至 2021/03/13\
    >testing dara的範圍從 2019/01/02 至 2021/03/21

    該處理方式為Prophet所設計，其輸入一段時間的資料 (**包含日期及資料**) 後，\
    可輸出後8天的預測資料。\
    同樣各資料的維度列表如下：
    |            | Training | Validation | Testing |
    |------------|----------|------------|---------|
    | Data Dim.  |          | (795,3)    | (810,3) |
    | Label Dim. |          | (8)        |         |

## 模型

依使用套件的種類可分作下列幾種：

### Sklearn

* `python app.py --training training_data.csv --output submission.csv --model sklearn`

* 詳細模型參數以及架構如下，為一般的MLP：

  ```python
  model = MLPRegressor(random_state=1,
                      hidden_layer_sizes=(2),
                      activation="relu",
                      solver='adam',
                      batch_size=16,
                      learning_rate="constant",
                      learning_rate_init=1e-3,
                      max_iter=1000)
  ```

* 使用validation data (2021/01/21 至 2021/02/19) 之預測值與實際值之RMSE為：\
\
  **RMSE = 116.3423**

  ![SK_RMSE](https://i.imgur.com/UsI0luI.png)

* 最終輸出檔名為：`sklearn_submission.csv`

### Pytorch

* `python app.py --training training_data.csv --output submission.csv --model pytorch`

* 同樣為一般的MLP架構，但其可變性較Sklearn高，\
  亦可看見其訓練過程，並亦可使用GPU加速。\
  然而此處我們省略了將每個epoch的loss print出來的部分。\
  \
  架構如下：
  ```python
  class m02(nn.Module):
      def __init__(self, in_num, seq):
          super(m02, self).__init__()
          self.FC = nn.Sequential(
              nn.Flatten(),
              nn.Linear(in_num*seq, 64),
              nn.ReLU(),
              nn.Linear(64, 8)
              )
      def forward(self, x):
          pred = self.FC(x)
          return pred
  ```

  參數如下：

  ```python
  batch = 16
  lr = 1e-3
  Epoch = 1000
  model = m02(3, 30)
  loss_F = nn.MSELoss()
  optim = optim.Adam(model.parameters(), lr=lr)
  ```

* 使用validation data (2021/01/21 至 2021/02/19) 之預測值與實際值之RMSE為：\
\
  **RMSE = 110.0655**

  ![PT_RMSE](https://i.imgur.com/PFsdD4X.png)

* 最終輸出檔名為：`pytorch_submission.csv`

### Prophet

[官方網站](https://facebook.github.io/prophet/)

* `python app.py --training training_data.csv --output submission.csv --model prophet`

* 安裝步驟 (python 3.6.4)

  * Windows 10

      1. 安裝Microsoft C++ Build Tools \
          [下載連結](https://visualstudio.microsoft.com/zh-hant/visual-cpp-build-tools/)\
          僅勾選"C++建置工具即可"

          ![pic](https://i.imgur.com/s7YbNq4.png)

      2. `pip install pystan==2.17.1.0 fbprophet==0.6`

  * Ubuntu 16.04.3 LTS
      1. `apt-get update`

      2. `apt-get install build-essential`

      3. `pip install pystan==2.17.1.0 fbprophet==0.6`

* 此處的使用方式為，輸入過去數天的日期以及對應的數值，\
  選擇輸出後幾天 (此處統一設定為後14天) 的數據，\
  方可預測完成。

  ```python
  val_pred = forecastByProphet(VAL_data2, 8)
  # VAL_data2: (time by data)
  # val_pred: result of pred. (8)
  ```

* 使用validation data (2019/01/02 至 2021/03/13) 之預測值與實際值之RMSE為：\
\
  **RMSE = 98.4839**

  ![PR_RMSE](https://i.imgur.com/xNUeRks.png)

* 最終輸出檔名為：`prophet_submission.csv`

### 小節

從上述三張驗證集的預測軌跡看來，以**Prophet最為理想**，\
故本次作業我們選用該預測結果當作最終答案，\
即檔名為：`submission.csv`

>實際上在此處我們使用了三個模型進行預測，\
>若僅為了提高準確度，應可從資料中sample出三群dataset，\
>進行預測之後再使用bagging，\
>應可得到variance與bias皆較小的預測值。

>另一方面，目前使用Prophet可得到長期趨勢，\
>以及其他週期性方程式，未來若夠以Prophet預測趨勢，\
>並以MLP等模型來預測其餘非線性的部分，\
>說不定可以讓誤差進一步縮小
