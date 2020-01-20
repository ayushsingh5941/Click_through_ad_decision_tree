import pandas as pd
file = pd.read_csv('test.csv', nrows=100000)
file.to_csv(r'test1.csv')