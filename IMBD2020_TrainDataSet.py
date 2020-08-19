import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 設定 data_path
dir_data = r'D:\109919_Source'
# dir_data = 'D:\ML\IMBD2020'
raw_data_train = os.path.join(dir_data, '0714train.csv')
# f_app_train = os.path.join(dir_data, '0714train_XY.csv')
f_app_test = os.path.join(dir_data, "0728test.csv")
# 讀取檔案
app_train = pd.read_csv(raw_data_train)
app_test = pd.read_csv(f_app_test)
raw_train = pd.read_csv(raw_data_train)

raw_trainx = raw_train.iloc[:, :282]
raw_trainy = raw_train.iloc[:, 282:]
train_x = app_train
test_x = app_test

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

col_x = train_x.columns
# col_y = train_y.columns
A1_x = ["Number"]
A2_x = ["Number"]
A3_x = ["Number"]

A6_x = ["Number"]
A1_y = []
A2_y = []
A3_y = []

A6_y = []
C_y = []
test_A1 = ["Number"]
test_A2 = ["Number"]
test_A3 = ["Number"]
test_A6 = ["Number"]
test_C = ["Number"]

x_A1 = pd.DataFrame()
x_A2 = pd.DataFrame()
x_A3 = pd.DataFrame()
x_A6 = pd.DataFrame()
y_A1 = pd.DataFrame()
y_A2 = pd.DataFrame()
y_A3 = pd.DataFrame()
y_A6 = pd.DataFrame()
y_C = pd.DataFrame()
testX_A1 = pd.DataFrame()
testX_A2 = pd.DataFrame()
testX_A3 = pd.DataFrame()
testX_A6 = pd.DataFrame()
testX_C = pd.DataFrame()


df_temp = pd.DataFrame()
inputC_x = 0
inputC_y = 0
inputC_X = []
inputC_Y = []
testC_x = 0
testC_y = 0



for i in train_x.columns:
    inputC_X = []
    inputC_Y = []
    if np.dtype(train_x[i]) == object and i != "Number":
        for n, j in enumerate(train_x[i]):
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
        train_x[df_columnsX] = inputC_X
        train_x[df_columnsY] = inputC_Y
        raw_trainx[df_columnsX] = inputC_X
        raw_trainx[df_columnsY] = inputC_Y
        train_x = train_x.drop([i], axis=1)
        raw_trainx = raw_trainx.drop([i], axis=1)

'''
以下，功能為訓練模型預測feature缺值：
1. Feature_trainX是一個去除有缺值資料集的資料，因此Feature_trainX為無缺值data set。
2. null_list是一個缺值數命名的字典，內部values是dataframe, 每一個dataframe均為相同缺值個數的data set。
3. 依據feature所缺之值個數，拆分成字典GG_dict_number, number代表著所缺的feature數量。
4. 每一個GG_dict_number內的dataframe均為模型預測缺值的測試集(Test set)。
'''
null_df = pd.DataFrame()
null_values = []
null_index = []

x = train_x.isnull().sum(axis=0)
for v, i in zip(x.values, x.index):
    if v > 0:
        null_values.append(v)
        null_index.append(i)
null_df["null_name"] = null_index
null_df["null_values"] = null_values
null_df = train_x[train_x.isnull().values==True].copy()
null_df = null_df.drop_duplicates(subset=['Number'], keep='first')

Feature_test2 = pd.DataFrame()
Feature_test8 = pd.DataFrame()
Feature_test9 = pd.DataFrame()
Feature_test10 = pd.DataFrame()
Feature_test16 = pd.DataFrame()
Feature_test24 = pd.DataFrame()
Feature_test30 = pd.DataFrame()
Feature_test32 = pd.DataFrame()
Feature_test42 = pd.DataFrame()
Feature_test49 = pd.DataFrame()
Feature_test51 = pd.DataFrame()
Feature_test89 = pd.DataFrame()
Feature_test120 = pd.DataFrame()

Feature_trainX = train_x.copy()
index_null = null_df.index
Feature_trainX = Feature_trainX.drop(index_null)


null_list = {2:Feature_test2, 8:Feature_test8, 9:Feature_test9, 10:Feature_test10, 16:Feature_test16,
             24:Feature_test24, 30:Feature_test30, 32:Feature_test32, 42:Feature_test42, 49:Feature_test49,
             51:Feature_test51, 89:Feature_test89, 120:Feature_test120}

for n, i in enumerate(null_df["Number"]):
    x = null_df[null_df["Number"]==i]
    for L in null_list:
        if np.sum(x.isnull().sum(axis=0)) == L:
            null_list[L] = null_list[L].append(null_df[null_df["Number"]==i], ignore_index=True)

# ===================================================================================================

GG_list2 = []
GG_dict2 = {}
GG_list8 = []
GG_dict8 = {}
GG_list9 = []
GG_dict9 = {}
GG_list10 = []
GG_dict10 = {}
GG_list16 = []
GG_dict16 = {}
GG_list24 = []
GG_dict24 = {}
GG_list30 = []
GG_dict30 = {}
GG_list32 = []
GG_dict32 = {}
GG_list42 = []
GG_dict42 = {}
GG_list49 = []
GG_dict49 = {}
GG_list51 = []
GG_dict51 = {}
GG_list89 = []
GG_dict89 = {}
GG_list120 = []
GG_dict120 = {}
predict_X = null_list[2]
predict_X27 = null_list[2]
predict_X29 = null_list[2]
predict_X8 = null_list[8]
predict_X9 = null_list[9]
predict_X10 = null_list[10]
predict_X16 = null_list[16]
predict_X24 = null_list[24]
predict_X30 = null_list[30]
predict_X32 = null_list[32]
predict_X42 = null_list[42]
predict_X49 = null_list[49]
predict_X51 = null_list[51]
predict_X89 = null_list[89]
predict_X120 = null_list[120]


predict_dict = {8:predict_X8, 9:predict_X9, 10:predict_X10, 16:predict_X16,
                24:predict_X24, 30:predict_X30, 32:predict_X32, 42:predict_X42,
                49:predict_X49, 51:predict_X51, 89:predict_X89, 120:predict_X120}
for i in predict_dict:
    x = predict_dict[i].isnull().sum()

for null_list_key in null_list:
    xx = null_list[null_list_key].isnull().sum()
    yy = null_list[null_list_key]
    if null_list_key == 2:
        for i in range(len(yy.index)):
            GG_list2.append(i)
            GG_list2[i] = pd.DataFrame()
            GG_dict2[i] = GG_list2[i]
    elif null_list_key == 8:
        for i in range(len(yy.index)):
            GG_list8.append(i)
            GG_list8[i] = pd.DataFrame()
            GG_dict8[i] = GG_list8[i]
    elif null_list_key == 9:
        for i in range(len(yy.index)):
            GG_list9.append(i)
            GG_list9[i] = pd.DataFrame()
            GG_dict9[i] = GG_list9[i]
    elif null_list_key == 10:
        for i in range(len(yy.index)):
            GG_list10.append(i)
            GG_list10[i] = pd.DataFrame()
            GG_dict10[i] = GG_list10[i]
    elif null_list_key == 16:
        for i in range(len(yy.index)):
            GG_list16.append(i)
            GG_list16[i] = pd.DataFrame()
            GG_dict16[i] = GG_list16[i]
    elif null_list_key == 24:
        for i in range(len(yy.index)):
            GG_list24.append(i)
            GG_list24[i] = pd.DataFrame()
            GG_dict24[i] = GG_list24[i]
    elif null_list_key == 30:
        for i in range(len(yy.index)):
            GG_list30.append(i)
            GG_list30[i] = pd.DataFrame()
            GG_dict30[i] = GG_list30[i]
    elif null_list_key == 32:
        for i in range(len(yy.index)):
            GG_list32.append(i)
            GG_list32[i] = pd.DataFrame()
            GG_dict32[i] = GG_list32[i]
    elif null_list_key == 42:
        for i in range(len(yy.index)):
            GG_list42.append(i)
            GG_list42[i] = pd.DataFrame()
            GG_dict42[i] = GG_list42[i]
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
    elif null_list_key == 89:
        for i in range(len(yy.index)):
            GG_list89.append(i)
            GG_list89[i] = pd.DataFrame()
            GG_dict89[i] = GG_list89[i]
    elif null_list_key == 120:
        for i in range(len(yy.index)):
            GG_list120.append(i)
            GG_list120[i] = pd.DataFrame()
            GG_dict120[i] = GG_list120[i]
    for y_Index in yy.index:
        for n, x_index in enumerate(xx[xx.values > 0].index):
            if np.isnan(yy.loc[y_Index, x_index]):
                if null_list_key == 2:
                    GG_dict2[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict2[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                    if y_Index ==0:
                        predict_X27 = predict_X27[predict_X27.index==y_Index]
                        predict_X27 = predict_X27.drop(x_index, axis=1)
                    elif y_Index ==1:
                        predict_X29 = predict_X29[predict_X29.index==y_Index]
                        predict_X29 = predict_X29.drop(x_index, axis=1)
                elif null_list_key == 8:
                    GG_dict8[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict8[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 10:
                    GG_dict8[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict8[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 16:
                    GG_dict16[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict16[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 24:
                    GG_dict24[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict24[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 30:
                    GG_dict30[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict30[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 32:
                    GG_dict32[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict32[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 42:
                    GG_dict42[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict42[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 49:
                    GG_dict49[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict49[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 51:
                    GG_dict51[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict51[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 89:
                    GG_dict89[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict89[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]
                elif null_list_key == 120:
                    GG_dict120[y_Index].loc[y_Index, "Number"] = yy.loc[y_Index, "Number"]
                    GG_dict120[y_Index].loc[y_Index, x_index] = yy.loc[y_Index, x_index]

# ========================================================================


'''
*****
======
AAAAA
'''

'''
預測缺值個數為2的數值
'''
estimator = RandomForestRegressor(n_estimators=150, max_depth=8)

predict2 = {0:predict_X27, 1:predict_X29}
for n, i in enumerate(GG_dict2):
    null2_trainX_1 = Feature_trainX.copy()
    null2_testX_1 = pd.DataFrame()
    for j in GG_dict2[i].columns:
        if j != "Number":
            null2_testX_1[j] = null2_trainX_1[j]
            null2_trainX_1 = null2_trainX_1.drop(j, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(null2_trainX_1.loc[:, "Input_A1_001":], null2_testX_1, test_size=0.1, random_state=4)
    estimator.fit(x_train, y_train)

    y_pred = estimator.predict(predict2[n].loc[:, "Input_A1_001":])#predict
    # y_pred = estimator.predict(x_test) #check mse

    x = y_pred[0]
    if n == 0:
        null_list[2].loc[0, "Input_C_027_x"] = x[0]
        null_list[2].loc[0, "Input_C_027_y"] = x[1]
        null_list[2] = null_list[2].reindex(columns=train_x.columns)
        Feature_trainX = Feature_trainX.append(null_list[2][null_list[2].index == 0], ignore_index=True)
    if n == 1:
        null_list[2].loc[1, "Input_C_029_x"] = x[0]
        null_list[2].loc[1, "Input_C_029_y"] = x[1]
        null_list[2] = null_list[2].reindex(columns=train_x.columns)
        Feature_trainX = Feature_trainX.append(null_list[2][null_list[2].index == 1], ignore_index=True)

'''
預測其他個數的缺值
'''



null_list_check = [8, 9, 10, 16, 24, 30, 32, 42, 49, 51, 89, 120]

y_pred = []
for null_list_key in predict_dict:
    trainX = Feature_trainX.copy()
    testX = pd.DataFrame()
    N = predict_dict[null_list_key].isnull().sum()
    if null_list_key != 2:
        for drop in N[N.values > 0].index:
            predict_dict[null_list_key] = predict_dict[null_list_key].drop(drop, axis=1)
    for i in null_list_check:
        if null_list_key == i:
            for col in N[N.values > 0].index:
                testX[col] = trainX[col]
                trainX = trainX.drop(col, axis=1)

            if null_list_key == 42:
                A1 = []
                A2 = []
                A3 = []
                A4 = []
                A5 = []
                A6 = []
                C = []
                testA1 = []
                testA2 = []
                testA3 = []
                testA4 = []
                testA5 = []
                testA6 = []
                testC = []
                train_x1 = pd.DataFrame()
                train_x2 = pd.DataFrame()
                train_x3 = pd.DataFrame()
                train_x4 = pd.DataFrame()
                train_x5 = pd.DataFrame()
                train_x6 = pd.DataFrame()
                train_Cx = pd.DataFrame()
                train_y1 = pd.DataFrame()
                train_y2 = pd.DataFrame()
                train_y3 = pd.DataFrame()
                train_y4 = pd.DataFrame()
                train_y5 = pd.DataFrame()
                train_y6 = pd.DataFrame()
                train_Cy = pd.DataFrame()
                test_x1 = pd.DataFrame()
                test_x2 = pd.DataFrame()
                test_x3 = pd.DataFrame()
                test_x4 = pd.DataFrame()
                test_x5 = pd.DataFrame()
                test_x6 = pd.DataFrame()
                test_C6 = pd.DataFrame()
                for feature in predict_dict[null_list_key].columns:
                    if "A1" in feature:
                        A1.append(feature)
                    elif "A2" in feature:
                        A2.append(feature)
                    elif "A3" in feature:
                        A3.append(feature)
                    elif "A4" in feature:
                        A4.append(feature)
                    elif "A5" in feature:
                        A5.append(feature)
                    elif "A6" in feature:
                        A6.append(feature)
                    if "C" in feature:
                        A1.append(feature)
                        A2.append(feature)
                        A3.append(feature)
                        A4.append(feature)
                        A5.append(feature)
                        A5.append(feature)
                        C.append(feature)
                for test_feature in testX.columns:
                    if "A1" in test_feature:
                        testA1.append(test_feature)
                    if "A2" in test_feature:
                        testA2.append(test_feature)
                    if "A3" in test_feature:
                        testA3.append(test_feature)
                    if "A4" in test_feature:
                        testA4.append(test_feature)
                    if "A5" in test_feature:
                        testA5.append(test_feature)
                    if "A6" in test_feature:
                        testA6.append(test_feature)
                    if "C" in test_feature:
                        testC.append(test_feature)
                for feature in A1:
                    train_x1["Number"] = trainX["Number"]
                    train_x1[feature] = trainX[feature]
                    test_x1[feature] = predict_dict[null_list_key][feature]
                for featureY in testA1:
                    train_y1[featureY] = testX[featureY]
                for feature in A2:
                    train_x2["Number"] = trainX["Number"]
                    train_x2[feature] = trainX[feature]
                    test_x2[feature] = predict_dict[null_list_key][feature]
                for featureY in testA2:
                    train_y2[featureY] = testX[featureY]
                for feature in A3:
                    train_x3["Number"] = trainX["Number"]
                    train_x3[feature] = trainX[feature]
                    test_x3[feature] = predict_dict[null_list_key][feature]
                for featureY in testA3:
                    train_y3[featureY] = testX[featureY]
                for feature in A4:
                    train_x4["Number"] = trainX["Number"]
                    train_x4[feature] = trainX[feature]
                    test_x4[feature] = predict_dict[null_list_key][feature]
                for featureY in testA4:
                    train_y4[featureY] = testX[featureY]
                for feature in A5:
                    train_x5["Number"] = trainX["Number"]
                    train_x5[feature] = trainX[feature]
                    test_x5[feature] = predict_dict[null_list_key][feature]
                for featureY in testA5:
                    train_y5[featureY] = testX[featureY]
                for feature in A6:
                    train_x6["Number"] = trainX["Number"]
                    train_x6[feature] = trainX[feature]
                    test_x6[feature] = predict_dict[null_list_key][feature]
                for featureY in testA6:
                    train_y6[featureY] = testX[featureY]
                for feature in C:
                    train_Cx["Number"] = trainX["Number"]
                    train_Cx[feature] = trainX[feature]
                    test_C6[feature] = predict_dict[null_list_key][feature]
                for featureY in testC:
                    train_Cy[featureY] = testX[featureY]

                x_train1, x_test1, y_train1, y_test1 = train_test_split(train_x1.loc[:, "Input_A1_001":], train_y1,
                                                                    test_size=0.1, random_state=4)
                estimator.fit(x_train1, y_train1)
                y_pred1 = estimator.predict(test_x1) #predict
                # y_pred1 = estimator.predict(x_test1) #check mse


                x_train2, x_test2, y_train2, y_test2 = train_test_split(train_x2.loc[:, "Input_A2_001":], train_y2,
                                                                    test_size=0.1, random_state=4)

                estimator.fit(x_train2, y_train2)
                y_pred2 = estimator.predict(test_x2.loc[:, "Input_A2_001":])#predict
                # y_pred2 = estimator.predict(x_test2) #check mse


                x_train3, x_test3, y_train3, y_test3 = train_test_split(train_x3.loc[:, "Input_A3_001":], train_y3,
                                                                        test_size=0.1, random_state=4)
                estimator.fit(x_train3, y_train3)
                y_pred3 = estimator.predict(test_x3.loc[:, "Input_A3_001":])#predict
                # y_pred3 = estimator.predict(x_test3) #check mse

                x_train4, x_test4, y_train4, y_test4 = train_test_split(train_x4.loc[:, "Input_A4_001":], train_y4,
                                                                        test_size=0.1, random_state=4)
                estimator.fit(x_train4, y_train4)
                y_pred4 = estimator.predict(test_x4.loc[:, "Input_A4_001":])#predict
                # y_pred4 = estimator.predict(x_test4) #check mse

                x_train5, x_test5, y_train5, y_test5 = train_test_split(train_x5.loc[:, "Input_A5_001":], train_y5,
                                                                        test_size=0.1, random_state=4)
                estimator.fit(x_train5, y_train5)
                y_pred5 = estimator.predict(test_x5.loc[:, "Input_A5_001":])#predict
                # y_pred5 = estimator.predict(x_test5) #check mse

                x_train6, x_test6, y_train6, y_test6 = train_test_split(train_x6.loc[:, "Input_A6_001":], train_y6,
                                                                        test_size=0.1, random_state=4)
                estimator.fit(x_train6, y_train6)
                y_pred6 = estimator.predict(test_x6.loc[:, "Input_A6_001":])#predict
                # y_pred6 = estimator.predict(x_test6) #check mse

                y_PRED = [y_pred1, y_pred2, y_pred3, y_pred4, y_pred5, y_pred6]
                train_list = [train_y1, train_y2, train_y3, train_y4, train_y5, train_y6]
                test_list = [test_x1, test_x2, test_x3, test_x4, test_x5, test_x6]

                for Y, i in enumerate(y_PRED):
                    for test_Index in test_list[Y].index:

                        x = i[test_Index]
                        for n, col in zip(range(len(x)), train_list[Y].columns):
                            null_list[null_list_key].loc[test_Index, col] = x[n]
                            null_list[null_list_key] = null_list[null_list_key].reindex(columns=train_x.columns)
                Feature_trainX = Feature_trainX.append(null_list[null_list_key], ignore_index=True)

            else:
                x_train, x_test, y_train, y_test = train_test_split(trainX.loc[:, "Input_A1_001":], testX, test_size=0.1, random_state=4)
                estimator.fit(x_train, y_train)
                y_pred = estimator.predict(predict_dict[null_list_key].loc[:, "Input_A1_001":])#predict
                # y_pred = estimator.predict(x_test) #check mse
                # print("null_list_key %i:" %null_list_key, mse(y_test, y_pred))

                for Index in null_list[null_list_key].index:
                    x = y_pred[Index]
                    for n, col in zip(range(len(x)), N[N.values > 0].index):

                        null_list[null_list_key].loc[Index, col] = x[n]
                        null_list[null_list_key] = null_list[null_list_key].reindex(columns=train_x.columns)
                Feature_trainX = Feature_trainX.append(null_list[null_list_key], ignore_index=True)
                x = Feature_trainX.isnull().sum()
    print(Feature_trainX.shape)
Feature_trainX.to_csv("train_X.csv", index=False)
# Feature_trainX.to_csv(r"D:\ML\IMBD2020\train_X.csv", index = False)

# print(Feature_trainX.isnull().sum())


