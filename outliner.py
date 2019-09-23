0000000000# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 23:20:24 2019

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:06:26 2019

@author: lenovo
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import load_boston

boston = load_boston()

x = boston.data
x
y = boston.target
y
columns = boston.feature_names
columns
boston

### Create the dataframe

boston_df = pd.DataFrame(boston.data)

boston_df.columns = columns

boston_df.head()

### Box plot 

sns.boxplot(x = boston_df['DIS'])

### Z score  function  defined

from scipy import stats
import numpy as np

z = np.abs(stats.zscore(boston_df))
z
## formula for z score = (observation - Mean)/ standard deviation
## Z table always have the threshold of  -3 and 3. Any value greater than 3 it is any outliner


print(np.where(z>3))

print(z[105][6]) 
### The first array contains the row number and second array respective column number

Q1 = boston_df.quantile(0.25)
Q3 = boston_df.quantile(.75)
IQR = Q3 - Q1
print(IQR)

print((boston_df<(Q1-1.5*IQR))|(boston_df>(Q3+1.5*IQR)))

### Datapoint where we have false that means these values are valid. 
###True indicates presence of outliners

boston_df = boston_df[(z<3).all(axis =1)]

boston_df.shape








