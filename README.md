# IMDB2020
題目：「加工機台參數預測」
以加工機台完整的「加工參數」和「加工品質」作為訓練資料，於測試階段預
測 20項重點參數。

![image](file://D:/ML/IMDB%E6%B5%81%E7%A8%8B%E5%9C%96/readme.JPG)


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
1. 開啟Pycharm，並確認Python為3.7版本，以及其他套件在"109919_Source"資料夾下的requirments.txt檔，該文字檔內有所需套件的名稱以及版本。
2. 確認路徑"D:\109919_Source"下有0714train以及0728test兩份CSV檔。
3. 確認第一與第二個步驟均已滿足。
4. 先執行TrainDataSet資料夾內的IMBD2020_TrainDataSet.py，並確認train_X的CSV檔產生在路徑"D:\IMBD2020\TrainDataSet"。
5. 確認路徑"D:\109919_Source"有train_X以及0728test兩份CSV檔。
6. 再執行PredictResult資料夾內的IMBD2020_PredictResult.py檔，並確認IMBD2020_TestResult的CSV檔已產生。
7. IMBD2020_TestResult為預測結果的CSV檔。

