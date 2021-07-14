'''EDA of the Dataset'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


'Prediction CO'

#read data
df = pd.read_csv('./data/train.csv')

#inspect data
df.head()
df.tail()

delta_ts = df.index[-1] - df.index[0]
print('The differce of the first datapoint and last is ', delta_ts)

df.info()
df.describe()

pd.isnull(df).sum() > 0

df.columns

#Clean df
df.drop(['target_benzene','target_nitrogen_oxides'], axis=1,inplace=True)

#set index as DateTime
df.set_index(['date_time'], inplace=True)
df.index = pd.to_datetime(df.index)
df['h'] = df.index.hour

#heatmap 
plt.figure(figsize=(10,8))
sns.heatmap(round(df.corr(),2), annot=True)
plt.savefig('./heatmap.png', dpi=150)
plt.show()

#train_test_split
plt.figure(figsize=(10,8))
plt.plot(df.index, df['target_carbon_monoxide'])

df_test = df.iloc[1500:2000].append(df.iloc[6000:6500])
df_train = df.iloc[0:1499].append(df.iloc[2000:5999])
df_train = df_train.append(df.iloc[6500:-1])



X_CO = df_train[['sensor_1','sensor_2','sensor_3','sensor_4','sensor_5','h']]
y_CO = df_train['target_carbon_monoxide']

plt.figure(figsize=(10,8))
plt.scatter(df_train.index, df_train['target_carbon_monoxide'])

pd.isnull(df_train.index).sum() > 0

#Analyze Trend
df_train['timestep'] = range(len(df_train))
Xtrend = df_train[['timestep']]

m = LinearRegression()
m.fit(Xtrend, y_CO)

X_CO['trend'] = m.predict(Xtrend)
plt.figure(figsize=(20,15))
plt.plot(df_train.index, df_train['target_carbon_monoxide'])
plt.plot(df_train.index, X_CO['trend'])

#Analyze Seasonality
X_CO['day'] = X_CO.index.day
seasonal_dummies = pd.get_dummies(X_CO['day'], prefix='day', drop_first=True)

X_CO = X_CO.merge(seasonal_dummies, left_index = True, right_index = True)
Xseason = X_CO.drop(['sensor_1','sensor_2','sensor_3','sensor_4','sensor_5','h','trend'],axis=1)
m_season = LinearRegression()
m_season.fit(Xseason, y_CO)

X_CO['seasonANDtrend'] = m_season.predict(Xseason)

plt.figure(figsize=(20,15))
plt.plot(df_train.index, df_train['target_carbon_monoxide'])
plt.plot(df_train.index, X_CO['seasonANDtrend'])

#Analyze Remainder

X_CO['remainder'] = df_train['target_carbon_monoxide'] - X_CO['seasonANDtrend']
plt.figure(figsize=(20,15))
plt.plot(df_train.index, X_CO['remainder'])
plt.plot(df_train.index, df_train['target_carbon_monoxide'], alpha=0.5)
plt.legend(['remainder','CO'])


