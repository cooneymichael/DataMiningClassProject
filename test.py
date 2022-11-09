import pandas as pd
import decisiontree as dt
import decisiontreetwo as dt2

data = pd.read_csv('./mutations.csv', index_col='Unnamed: 0')

# decision_tree = dt.DecisionTreePhi(data=data, depth=2)
# decision_tree = dt.DecisionTreeGain(data = data, depth = 3)
# decision_tree = dt.DecisionTree(data, 2)

decision_tree = dt2.DecisionTreePhi(data=data, depth=3)

print(decision_tree)
