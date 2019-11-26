# normalizes data
import pandas as pd
import numpy as np

file = "data/disk.csv"

df = pd.read_csv(file)

print(df.columns.values)

df['0'] = df['0']/np.linalg.norm(df['0'])
df['1'] = df['1']/np.linalg.norm(df['1'])

df.to_csv(file.replace("_normal","").replace(".csv", "_normal.csv"), index=False)

