import pandas as pd

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:,1:2].values
headset = dataset.head()
describe = dataset.describe()
rows = len(dataset.index)
columns = len(dataset.columns)



print (describe)
print('')
print (headset)
print('')
print(rows)
print('')
print(columns)
