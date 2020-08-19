# IMDB2020
題目：「加工機台參數預測」
以加工機台完整的「加工參數」和「加工品質」作為訓練資料，於測試階段預
測 20項重點參數。

![image](https://github.com/Gaprs/IMDB2020/blob/master/readme.JPG)


File folder name: TrainDataSet
說明:
IMBD2020_TrainDataSet.py為資料前處理的,py檔，會自動產生train_X的CSV檔，該檔案為資料清理完後的Data set。
在缺值填補的部分採用Random Forest的演算法進行預測缺值並填補，順序是從所缺的feature數最少的資料開始預測，每預測完就新增回無缺值的資料集並再次預測下一筆缺值。

File folder name: PredictResult
說明:
IMBD2020_PredictResult為根據上述產生的train_X CSV檔作為類神經網絡的輸入，
並透過類神經網絡預測結果並自動產出IMBD2020_TestResult的CSV檔作為預測結果。
類神經網絡建構為七層的hidden layer, Activation is LeakyReLU, 每層均採用Dropout, Optimizers is Adam, lr is 0.0001, beta_1=0.9, beta_2=0.95, decay=1e-6
epoch 1000 times, batch size 64.


執行步驟:
1. 執行TrainDataSet資料夾內的IMBD2020_TrainDataSet.py，並產生train_X的CSV檔為資料前處理後的資料集。
2. 再執行PredictResult資料夾內的IMBD2020_PredictResult.py檔，並產生IMBD2020_TestResult的CSV檔為預測結果的資料集。
3. IMBD2020_TestResult為預測結果的CSV檔。

