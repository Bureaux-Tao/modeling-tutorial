from utils import random_forest
import pandas as pd
import numpy as np

data = pd.DataFrame({'id': [1, np.nan, np.nan, np.nan, 5, 6], 'name': [22, np.nan, 12, 22, 377, 200],
                     'math': [90, np.nan, 99, 78, 97, np.nan], 'english': [89, np.nan, 80, 94, 94, 90]})

target = pd.DataFrame({'lable': [1, 0, 1, 1, 1, 0]})

print(data)

data = random_forest.fill_na(data, target, 50)

print(data)
