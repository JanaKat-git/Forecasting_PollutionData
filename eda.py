'''EDA of the Dataset'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read data
df = pd.read_csv('./data/train.csv')

#inspect data
df.info()
df.describe()

pd.isnull(df).sum() > 0

plt.figure(figsize=(10,8))
sns.heatmap(round(df.corr(),2), annot=True)
plt.savefig('./heatmap.png', dpi=150)
plt.show()

X_CO = df[['']]
y_CO = df['']