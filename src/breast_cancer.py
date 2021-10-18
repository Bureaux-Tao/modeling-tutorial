##
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

##
data = load_breast_cancer()
print(data.data.shape)
print(data.target)

##
rfc = RandomForestClassifier(n_estimators = 100, random_state = 90)
score_pre = cross_val_score(rfc, data.data, data.target, cv = 10).mean()

print(score_pre)

##
score1 = []
for i in range(0, 200, 10):
    print('n_estimators: ' + str(i))
    rfc = RandomForestClassifier(n_estimators = i + 1,
                                 n_jobs = -1,
                                 random_state = 90)
    score = cross_val_score(rfc, data.data, data.target, cv = 10).mean()
    score1.append(score)
# 获得交叉验证平均值中的最大值及其下标
# 因为是10个为单位取平均，所以下标要乘以10
print(max(score1), (score1.index(max(score1)) * 10) + 1)
plt.figure(figsize = [20, 5])
plt.plot(range(1, 201, 10), score1)
plt.show()

##
score2 = []
for i in range(71, 81):
    print('n_estimators: ' + str(i))
    rfc = RandomForestClassifier(n_estimators = i + 1,
                                 n_jobs = -1,
                                 random_state = 90)
    score = cross_val_score(rfc, data.data, data.target, cv = 10).mean()
    score2.append(score)
# 获得交叉验证平均值中的最大值及其下标
# 因为是10个为单位取平均，所以下标要乘以10
print(max(score2), (score2.index(max(score2)) * 10) + 1)
plt.figure(figsize = [20, 5])
plt.plot(range(71, 81), score2)
plt.show()

##
# 调整max_depth
param_grid = {'max_depth': np.arange(1, 20, 1)}
#   一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
#   但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
#   更应该画出学习曲线，来观察深度对模型的影响
rfc = RandomForestClassifier(n_estimators = 39,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 10)
GS.fit(data.data, data.target)
print(GS.best_params_)
print(GS.best_score_)

##
param_grid = {"max_features": np.arange(5, 30, 1)}
rfc = RandomForestClassifier(n_estimators = 39,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 10)
GS.fit(data.data, data.target)

print(GS.best_params_)
print(GS.best_score_)

##
param_grid = {'min_samples_leaf': np.arange(1, 1 + 10, 1)}
rfc = RandomForestClassifier(n_estimators = 39,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 10)
GS.fit(data.data, data.target)

print(GS.best_params_)
print(GS.best_score_)

##
param_grid = {'min_samples_split': np.arange(2, 2 + 20, 1)}
rfc = RandomForestClassifier(n_estimators = 39,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 10)
GS.fit(data.data, data.target)

print(GS.best_params_)
print(GS.best_score_)

##
param_grid = {'criterion': ['gini', 'entropy']}
rfc = RandomForestClassifier(n_estimators = 39,
                             random_state = 90)
GS = GridSearchCV(rfc, param_grid, cv = 10)
GS.fit(data.data, data.target)

print(GS.best_params_)
print(GS.best_score_)

##
rfc = RandomForestClassifier(n_estimators = 39, random_state = 90)
score = cross_val_score(rfc, data.data, data.target, cv = 10).mean()
print(score)
print(score - score_pre)
