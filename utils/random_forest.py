import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


def fill_na(data: pd.DataFrame, target: pd.DataFrame, n_estimators: int = 50):
    sortindex = data.isnull().sum().sort_values().index
    sortindex = list(sortindex)
    print('NaN value count rank: ' + str(sortindex))
    
    for i in sortindex:
        # 构建新特征和新标签
        df = data
        fillc = df.loc[:, i]
        df = pd.concat([df.loc[:, df.columns != i], target], axis = 1)
        # target 标签
        
        # 在新特征中，对有缺失值的列，用0填补 df_0：数组类型
        df_0 = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0).fit_transform(df)
        
        # 找出训练集和测试集
        # 找出当列非空的值
        Ytrain = fillc[fillc.notnull()]
        # 找出当列空的值
        Ytest = fillc[fillc.isnull()]
        # 根据非空值找出一整行数组
        Xtrain = df_0[Ytrain.index, :]
        # 根据空值找出一整行数组
        Xtest = df_0[Ytest.index, :]
        
        # ⽤随机森林回归来训练预测
        print('Filling ' + i + '...')
        rfc = RandomForestRegressor(n_estimators = n_estimators)
        rfc = rfc.fit(Xtrain, Ytrain)
        Ypredict = rfc.predict(Xtest)
        
        # 将填补好的特征返回到我们的原始的特征中
        data.loc[data.loc[:, i].isnull(), i] = Ypredict
    
    print('Complete')
    print(data.isnull().sum())
    return data
