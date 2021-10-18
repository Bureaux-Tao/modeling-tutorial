# Python数据建模教程

```
./
├── README.md
├── data                                               数据集
│   ├── PRSA_data_2010.1.1-2014.12.31.csv
│   ├── ccpp.csv
│   ├── fisher
│   │   ├── test_data.txt
│   │   └── train_data.txt
│   ├── flights.csv
│   ├── flights_export.csv
│   ├── international-airline-passengers.csv
│   ├── iris.txt
│   ├── jet_rail.csv
│   ├── portland-oregon-average-monthly.csv
│   ├── wine.data
│   └── 电影票房.csv
├── images                                             Jupyter所需图片
├── main.py
├── models                                             保存的模型
│   ├── international-airline-passengers.h5
│   └── multi_variables_regression.h5
├── notebooks                   
│   ├── Adaboost.ipynb                                 Adaboost算法
│   ├── DecisionTree.ipynb                             决策树
│   ├── DifferenceEquation.ipynb                       差分方程
│   ├── DifferentialEquation.ipynb                     微分方程
│   ├── EpidemicModel.ipynb                            传染病模型
│   ├── GraphTheory.ipynb                              图论
│   ├── GreyForecasting.ipynb                          灰色预测
│   ├── LinearProgramming.ipynb                        线性规划
│   ├── MonteCarloMethod.ipynb                         蒙特卡洛方法
│   ├── Networkx.ipynb                                 Networkx库的使用
│   ├── NumericalApproximation.ipynb                   数值逼近
│   ├── RandomForest.ipynb                             随机森林
│   ├── Regression.ipynb                               回归
│   └── TimeSequence.ipynb                             时间序列
├── requirements.txt                                   需要的python包
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── breast_cancer.py                               随机森林乳腺癌预测
│   ├── fill_na                                        随机森林填补空值
│   │   ├── __init__.py
│   │   ├── fill_na_1.py
│   │   └── fill_na_2.py
│   ├── fisher_analyse.py                              Fisher算法
│   ├── international_airline_passengers               时间序列乘客预测
│   │   ├── __init__.py
│   │   ├── international_airline_passengers.py
│   │   └── use_time_model.py
│   ├── interpolation.py                               插值
│   ├── multi_step_simension                           多维多步时间序列
│   │   ├── LSTM_Interface_Msteup.py
│   │   ├── LSTM_Interface_Msteup_II.py
│   │   ├── Test_Msteup.py
│   │   ├── Test_Msteup_II.py
│   │   └── __init__.py
│   ├── non_linear_regression.py                       非线性回归
│   ├── time_sequence
│   │   ├── __init__.py
│   │   ├── jet_rail.py
│   │   ├── multi-varibles-regression.py               多变量时间序列
│   │   └── slide_window.py                            滑动窗口时间序列预测
│   └── transportation_problem.py                      线性规划运输问题
└── utils                                              工具包
    ├── GM.py                                          GM算法
    ├── __init__.py
    ├── __pycache__
    ├── c_means_cluster.py                             c均值聚类
    ├── get_stability.py                               稳定性评价
    ├── grey_forecast.py                               灰色预测
    ├── random_forest.py                               随机森林填补空值
    └── series_to_supervised.py                        多维时间步转换

13 directories, 94 files
```