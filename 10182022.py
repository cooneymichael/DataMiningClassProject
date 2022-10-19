# coding: utf-8

import pandas as pd
data = pd.read_csv('mutations.csv', index_col="Unnamed: 0")
data
data.sample(230//3)
data.sample(230//3)
s1 = data.sample(230//3)
s1
s2 = data.drop(labels=s1.index).sample(230//3)
s2
s1
s3 = data.drop(labels=s1.index).drop(labels=s2.index)
s3
from decisionTree import DecisionTree as dt
get_ipython().run_line_magic('ls', '')
from decisiontree import DecisionTree as dt
training = s1.concat(s2)
training = s1.cat(s2)
help pandas
get_ipython().run_line_magic('help', 'pandas')
pd.concat([s1,s2])
trainging = pd.concat([s1,s2])
decision_tree = dt(testing)
decision_tree = dt(training)
del trainging
training = pd.concat([s1,s2])
decision_tree = dt(training)
decision_tree = dt(training, depth = 3)
print(decision_tree)
decision_tree = dt(training, depth = 2)
print(decision_tree)
testing = s3
for i in testing:
    print(i)
    
for i in testing.index:
    print(i)
    
    
conf_matrix
conf_matrix = {}
for i in testing.index:
    classification = decision_tree.classify(testing.loc[i, :])
    conf_matrix[i] = classification
    
conf_matrix
len(conf_matrix)
get_ipython().run_line_magic('save', '')
get_ipython().run_line_magic('save', '10182022.py')
get_ipython().run_line_magic('save', "'10182022.py'")
get_ipython().run_line_magic('save', '10182022.py 0-42')
conf_matrix
conf_matrix2 = {}
for i in conf_matrix:
    if i.startswith('NC'):
        conf_matrix2[i] = 'tn' if conf_matrix[i] == False else 'fp'
    else:
        conf_matrix2[i] = 'tp' if conf_matrix[i] == True else 'fn'
        
conf_matrix2
for i in range(len(conf_matrix)):
    print(i, conf_matrix[i], conf_matrix2[i])
    
for i in range(conf_matrix):
    print(i, conf_matrix[i], conf_matrix2[i])
    
for i in conf_matrix:
    print(i, conf_matrix[i], conf_matrix2[i])
    
get_ipython().run_line_magic('save', "-a '10182022.py' 43-50")
