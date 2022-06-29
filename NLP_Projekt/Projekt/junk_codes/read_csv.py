import sys
import pandas as pd


filename = sys.argv[1]

data= pd.read_csv(filename)

print(data.shape)