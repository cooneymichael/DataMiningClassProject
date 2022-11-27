import pandas as pd
import numpy as np
from decisiontree import *

data = pd.read_csv('mutations.csv', index_col='Unnamed: 0')
print(data)
