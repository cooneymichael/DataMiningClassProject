import pandas as pd
import decisiontree as dt

data = pd.read_csv('./mutations.csv', index_col='Unnamed: 0')

# decision_tree = dt.DecisionTreePhi(data=data, depth=2)
# decision_tree = dt.DecisionTreeGain(data = data, depth = 3)
decision_tree = dt.DecisionTree(data, 2)
print(decision_tree)
