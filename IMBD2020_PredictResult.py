import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import r2_score, accuracy_score
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 設定 data_path
dir_data = r'\109919_Source'
# dir_data = 'D:\ML\IMBD2020'
f_app_train = os.path.join(dir_data, 'train_X.csv')
f_app_test = os.path.join(dir_data, "0728test.csv")
# 讀取檔案
app_train = pd.read_csv(f_app_train)
app_test = pd.read_csv(f_app_test)

train_y = pd.DataFrame()
train_x = app_train
test_x = app_test

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


'''
將file name: 0728test.csv, 拆分成train data set and test data set.
將feature name Input_C_061, Input_C_062, Input_C_104 and Input_C_113用MaxAbsScaler標準化
'''

test_list = ['Input_A6_024', 'Input_A3_016', 'Input_C_013', 'Input_A2_016',
             'Input_A3_017', 'Input_C_050', 'Input_A6_001', 'Input_C_096',
             'Input_A3_018', 'Input_A6_019', 'Input_A1_020', 'Input_A6_011',
             'Input_A3_015', 'Input_C_046', 'Input_C_049', 'Input_A2_024',
             'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017']

'''
train_x 為整理好的trainX file裡的訓練集資料
train_y 為整理好的trainX file裡的測試集資料
train_x and train_y 用來train NN model
'''
for predict_list in test_list:
    train_y[predict_list] = app_train[predict_list]
    train_x = train_x.drop(predict_list, axis=1)


MMS = MinMaxScaler()
STD = StandardScaler()
MAS = MaxAbsScaler()
train_x["Input_C_061"] = MAS.fit_transform(pd.DataFrame(train_x["Input_C_061"]))
train_x["Input_C_062"] = MAS.fit_transform(pd.DataFrame(train_x["Input_C_062"]))
train_x["Input_C_113"] = MAS.fit_transform(pd.DataFrame(train_x["Input_C_113"]))
train_x["Input_C_104"] = MAS.fit_transform(pd.DataFrame(train_x["Input_C_104"]))

train_x1 = pd.DataFrame()
train_y1 = pd.DataFrame()
train_x1["Number"] = train_x["Number"]
for i in train_x.columns:
    if "A3" in i:
        train_x1[i] = train_x[i]
    if "C" in i:
        train_x1[i] = train_x[i]
for j in train_y.columns:
    if "A3" in j:
        train_y1[j] = train_y[j]
'''
整理 test data set
最後一行的 Feature_trainX 是整理好的test data set
用來predict result
test data set的缺值填補的方式與trainX file內填補的方式一樣，都是以RandomForest的方式從所缺feature數最少的開始預測
每預測完最少的data後都插入Feature_trainX data frame 的最後一列並重複上述步驟直到填補完所有缺值為止
'''
estimator = RandomForestRegressor(n_estimators=150, max_depth=8)

inputC_x = 0
inputC_y = 0
inputC_X = []
inputC_Y = []

'''
將input_C的偏移量feature根據XY軸拆分成兩個特徵，且根據XY軸的正負給定方向。
'''
for i in test_x.columns:
    inputC_X = []
    inputC_Y = []
    if np.dtype(test_x[i]) == object and i != "Number":
        for n, j in enumerate(test_x[i]):
            if type(j) == float:
                if np.isnan(j):
                    inputC_x = None
                    inputC_y = None
            else:
                intput_c = j.split(";")
                if intput_c[0] == "N":
                    inputC_y = 0
                elif intput_c[0] == "U":
                    inputC_y = np.float64(intput_c[1])
                elif intput_c[0] == "D":
                    inputC_y = -1 * np.float64(intput_c[1])
                if intput_c[2] == "N":
                    inputC_x = 0
                elif intput_c[2] == "R":
                    inputC_x = np.float64(intput_c[3])
                elif intput_c[2] == "L":
                    inputC_x = -1 * np.float64(intput_c[3])

            inputC_X.append(inputC_x)
            inputC_Y.append(inputC_y)
        df_columnsX = i + "_x"
        df_columnsY = i + "_y"
        test_x[df_columnsX] = inputC_X
        test_x[df_columnsY] = inputC_Y
        test_x = test_x.drop([i], axis=1)

null_df = pd.DataFrame()
null_values = []
null_index = []

x = test_x.isnull().sum(axis=0)
for v, i in zip(x.values, x.index):
    if v > 0:
        null_values.append(v)
        null_index.append(i)
null_df["null_name"] = null_index
null_df["null_values"] = null_values
null_df = test_x[test_x.isnull().values==True].copy()
null_df = null_df.drop_duplicates(subset=['Number'], keep='first')


'''
8, 49, 51各別代表著每筆資料所缺之feature數量
'''
Feature_test = pd.DataFrame()
Feature_test8 = pd.DataFrame()
Feature_test49 = pd.DataFrame()
Feature_test51 = pd.DataFrame()
null_list = {8:Feature_test8, 49:Feature_test49, 51:Feature_test51}

GG_list8 = []
GG_dict8 = {}
GG_list49 = []
GG_dict49 = {}
GG_list51 = []
GG_dict51 = {}


Feature_trainX = test_x.copy()
index_null = null_df.index
Feature_trainX = Feature_trainX.drop(index_null)

null_list_check = [8, 49, 51]

for n, i in enumerate(null_df["Number"]):
    x = null_df[null_df["Number"]==i]
    for L in null_list:
        if np.sum(x.isnull().sum(axis=0)) == L:
            null_list[L] = null_list[L].append(null_df[null_df["Number"] == i], ignore_index=True)

predict_X8 = null_list[8]
predict_X49 = null_list[49]
predict_X51 = null_list[51]
predict_dict = {8:predict_X8, 49:predict_X49, 51:predict_X51}

'''
以下功能為:將有缺值的資料抽取出來，並且按照缺值的feature數相同者同一組。
'''

for null_list_key in null_list:
    xx = null_list[null_list_key].isnull().sum()
    yy = null_list[null_list_key]

    if null_list_key == 8:
        for i in range(len(yy.index)):
            GG_list8.append(i)
            GG_list8[i] = pd.DataFrame()
            GG_dict8[i] = GG_list8[i]
    elif null_list_key == 49:
        for i in range(len(yy.index)):
            GG_list49.append(i)
            GG_list49[i] = pd.DataFrame()
            GG_dict49[i] = GG_list49[i]
    elif null_list_key == 51:
        for i in range(len(yy.index)):
            GG_list51.append(i)
            GG_list51[i] = pd.DataFrame()
            GG_dict51[i] = GG_list51[i]

    for y_Index in yy.index:
        for n, x_index in enumerate(xx[xx.values > 0].index):
            if np.isnan(yy.loc[y_Index, x_index]):
                if null_list_key == 8:
                    GG_dict8[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict8[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 49:
                    GG_dict49[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict49[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 51:
                    GG_dict51[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict51[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
'''
從缺少feature數最少的開始進行random forest預測缺值並填補
重複步驟直到全部填補完為止
'''
y_pred = []
for null_list_key in predict_dict:
    test_x = Feature_trainX.copy()
    testX = pd.DataFrame()

    N = predict_dict[null_list_key].isnull().sum()
    if null_list_key != 2:
        for drop in N[N.values > 0].index:
            predict_dict[null_list_key] = predict_dict[null_list_key].drop(drop, axis=1)

    for i in null_list_check:
        if null_list_key == i:
            for col in N[N.values > 0].index:
                testX[col] = test_x[col]
                test_x = test_x.drop(col, axis=1)

            x_train, x_test, y_train, y_test = train_test_split(test_x.loc[:, "Input_A1_001":], testX, test_size=0.1,
                                                                random_state=4)
            estimator.fit(x_train, y_train)
            y_pred = estimator.predict(predict_dict[null_list_key].loc[:, "Input_A1_001":])  # predict


            for Index in null_list[null_list_key].index:
                x = y_pred[Index]

                for n, col in zip(range(len(x)), N[N.values > 0].index):
                    null_list[null_list_key].loc[Index, col] = x[n]
                    null_list[null_list_key] = null_list[null_list_key].reindex(columns=Feature_trainX.columns)

            Feature_trainX = Feature_trainX.append(null_list[null_list_key], ignore_index=True)
            x = Feature_trainX.isnull().sum()

'''
由於train data set有針對下列四種feature做規一化
因此test data set也一樣
'''
Feature_trainX["Input_C_061"] = MAS.fit_transform(pd.DataFrame(Feature_trainX["Input_C_061"]))
Feature_trainX["Input_C_062"] = MAS.fit_transform(pd.DataFrame(Feature_trainX["Input_C_062"]))
Feature_trainX["Input_C_113"] = MAS.fit_transform(pd.DataFrame(Feature_trainX["Input_C_113"]))
Feature_trainX["Input_C_104"] = MAS.fit_transform(pd.DataFrame(Feature_trainX["Input_C_104"]))


'''
keras model fit, use earlystop
'''
earlystop = EarlyStopping(monitor="val_loss",
                          patience=50,
                          verbose=1,
                          mode='min'
                          )

'''
Create Model
1. function r2_score
2. function build_TrainModel, 建立具有七層隱藏層的類神經網絡，含Dropout, Adam, relu, softmax, lr=0.0001
以及使用mse and r2_score
3. 將訓練好的模型存入 file name : predict_Train_Model.h5

'''
Train_x, Test_x, Train_y, Test_y = train_test_split(train_x, train_y, test_size=0.4, random_state=32) #predict
# Train_x, Test_x, Train_y, Test_y = train_test_split(train_x1, train_y1, test_size=0.4, random_state=32) #test

def r2_score(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return K.ones_like(v) - (u / v)


def build_TrainModel(Train_x, Test_x, Train_y, Test_y):

    model = Sequential()
    model.add(Dense(128, input_dim=311))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(keras.layers.Dense(units=64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(keras.layers.Dense(units=64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(keras.layers.Dense(units=64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(keras.layers.Dense(units=32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(keras.layers.Dense(units=32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(keras.layers.Dense(units=32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(keras.layers.Dense(units=32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(keras.layers.Dense(20))
    # model.add(Dense(128, activation='relu', input_dim=311))
    # model.add(Dropout(0.3))
    # model.add(keras.layers.Dense(units=64, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(keras.layers.Dense(units=64, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(keras.layers.Dense(units=64, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(keras.layers.Dense(units=32, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(keras.layers.Dense(units=32, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(keras.layers.Dense(units=32, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(keras.layers.Dense(units=32, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(keras.layers.Dense(20))

    Adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.95, decay=1e-6)
    # sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(loss="mse",
                  optimizer=Adam,
                  metrics=[r2_score])
    model.fit(Train_x.loc[:, "Input_A1_001":], Train_y,
              epochs=1000,
              validation_data=(Test_x.loc[:, "Input_A1_001":], Test_y),
              batch_size=64,
              shuffle=True
              # callbacks=[earlystop]
              )
    # score = model.evaluate(xA1_test, yA1_test, batch_size=64)
    model.save('predict_Train_Model.h5')

def build_PredictModel(Test_x):

    model = load_model('predict_Train_Model.h5', custom_objects={'r2_score': r2_score})
    pred = pd.DataFrame(model.predict(Test_x.loc[:, "Input_A1_001":], batch_size=64))
    pred.columns = Test_y.columns
    pred.to_csv(r"\109919_Source\IMBD2020_TestResult.csv", index=False)
    # pred.to_csv(r"D:\ML\IMBD2020\TestResult_v1.csv", index=False)


build_TrainModel(Train_x, Test_x, Train_y, Test_y)
build_PredictModel(Feature_trainX)


