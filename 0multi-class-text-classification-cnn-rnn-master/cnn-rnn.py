
import pandas as pd
import os
for directories in os.listdir("data/"):
    print(directories)
print(os.path.exists("data/train.csv.zip"))

test_file = 'data/train.csv.zip'
df = pd.read_csv(test_file, sep='|')
select = ['Descript']
df = df.dropna(axis=0, how='any')
print(df.columns[0])
