'''Build Model to predict CO'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_validate
import numpy as np


#read data
df = pd.read_csv('./data/train.csv')

#set index as DateTime
df.set_index(['date_time'], inplace=True)
df.index = pd.to_datetime(df.index)
df['h'] = df.index.hour

#train-test-split
df_test = df.iloc[1500:2000].append(df.iloc[6000:6500])
df_train = df.iloc[0:1499].append(df.iloc[2000:5999])
df_train = df_train.append(df.iloc[6500:-1])

X_CO = df_train[['sensor_1','sensor_2','sensor_3','sensor_4','sensor_5','h']]
y_CO = np.log(df_train['target_carbon_monoxide']+1)

X_CO_test = df_test[['sensor_1','sensor_2','sensor_3','sensor_4','sensor_5','h']]
y_CO_test = np.log(df_test['target_carbon_monoxide']+1)


m = GradientBoostingRegressor(n_estimators=100, max_depth=5)
m.fit(X_CO, y_CO)
m.score(X_CO_test, y_CO_test)
df_train['prediction'] = m.predict(X_CO)

#Cross-validation
cross = cross_validate(m, X_CO, y_CO, cv = 5, return_train_score=True)
print('Cross-Validation test: '+str(round(cross['test_score'].mean(),2)) +' train: '+ str(round(cross['train_score'].mean(),2)))


def RMSLE(df):
    return np.sqrt(1/len(df)*(sum(np.log(df['prediction']+1) - np.log(df['target_carbon_monoxide']+1))**2))


display(df_train)

result = RMSLE(df_train)