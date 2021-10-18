##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.impute import SimpleImputer  # 填补缺失值的类
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

##
dataset = load_boston()
print(dataset.data.shape)
# 总共506*13=6578个数据
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

##
# 使50%数据缺失
rng = np.random.RandomState(0)
missing_rate = 0.5
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))  # np.floor向下取整，返回.0格式的浮点数

##
# randint(下限，上限，n)在上下限之间取出n个整数
missing_features = rng.randint(0, n_features, n_missing_samples)
missing_samples = rng.randint(0, n_samples, n_missing_samples)

##
missing_samples = rng.choice(n_samples, n_missing_samples)

X_missing = X_full.copy()
y_missing = y_full.copy()

# 可以直接通过列表形式赋值
X_missing[missing_samples, missing_features] = np.nan

print(X_missing.shape)

# 转换成DataFrame是为了后续方便各种操作，numpy对矩阵的运算速度快到拯救人生，但是在索引等功能上却不如 pandas来得好用
X_missing = pd.DataFrame(X_missing)
X_missing.head()

##
# 使用均值进行填补
# SimpleImputer是填补缺失值的类，strategy为填补数据的类型
imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')  # 实例化
X_missing_mean = imp_mean.fit_transform(X_missing)  # 训练fit+导出predict>>>特殊的接口fit_transform

# 使用0进行填补
imp_0 = SimpleImputer(missing_values = np.nan, strategy = "constant", fill_value = 0)
X_missing_0 = imp_0.fit_transform(X_missing)

# 查看是否还有空值，无空值，证明完全填补
pd.DataFrame(X_missing_mean).isnull().sum()

##
X_missing_reg = X_missing.copy()
# 找出数据集中，缺失值从小到大排列的特征的顺序，且得到这些特征的索引
# argsort返回从小到大排序的顺序所对应的索引
sortindex = np.argsort(X_missing_reg.isnull().sum(axis = 0)).values

print(sortindex)

for i in sortindex:
    # 构建我们的新特征矩阵（没有被选中去填充的特征+原始的标签）和新标签（被选中去填充的特征）
    df = X_missing_reg
    # 新标签
    fillc = df.iloc[:, i]
    # 新特征矩阵
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis = 1)
    
    # 在新特征矩阵中，对含有缺失值的列，进行0的填补
    df_0 = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 0).fit_transform(df)
    # 找出我们的训练集和测试集
    # 现在标签中存在的非空值
    Ytrain = fillc[fillc.notnull()]
    # 现在标签中的空值
    # 不需要Ytest的值，要的是Ytest的索引
    Ytest = fillc[fillc.isnull()]
    # 在新特征矩阵上，被选出来的要填充的非空值对应的记录
    Xtrain = df_0[Ytrain.index, :]
    # 新特征矩阵上，被选出来的要填充的那个特征的空值对应的记录
    Xtest = df_0[Ytest.index]
    
    # 用随机森林回归来填补缺失值
    rfc = RandomForestRegressor(n_estimators = 100)  # 实例化
    rfc = rfc.fit(Xtrain, Ytrain)  # 训练
    # 用predict接口将Xtest导入，得到预测结果作为填补空值的值
    Ypredict = rfc.predict(Xtest)
    
    # 将填补好的特征返回原始特征矩阵中
    X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), i] = Ypredict

##

# 对所有数据进行建模，取得MSE结果
X = [X_full, X_missing_mean, X_missing_0, X_missing_reg]

mse = []
std = []
for x in X:
    estimator = RandomForestRegressor(random_state = 0, n_estimators = 100)
    scores = cross_val_score(estimator, x, y_full, scoring = 'neg_mean_squared_error').mean()
    mse.append(scores * -1)

print([*zip(["X_full", "X_missing_mean", "X_missing_0", "X_missing_reg"], mse)])

##
x_labels = [
    'Full data'
    , 'Zero Imputation'
    , 'Mean Imputation'
    , 'Regressor Imputation'
]
colors = ['r', 'g', 'b', 'orange']

plt.figure(figsize = (12, 6))
ax = plt.subplot(111)
for i in np.arange(len(mse)):
    ax.barh(i, mse[i], color = colors[i], alpha = 0.6, align = 'center')
ax.set_title("Imputation Techniques with Boston Data")
ax.set_xlim(left = np.min(mse) * 0.9
            , right = np.max(mse) * 1.1)
ax.set_yticks(np.arange(len(mse)))
ax.set_xlabel('MSE')
ax.set_yticklabels(x_labels)
plt.savefig('compare_imputation', dpi = 300)
plt.show()
