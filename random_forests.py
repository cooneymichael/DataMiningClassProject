import pandas as pd
import numpy as np
from decisiontree import *
import copy
# from decisiontreetwo import DecisionTreePhi

NUMBER_OF_TREES = 2

def get_samples(index):
    return np.random.choice(index, size=index.size)

def get_features(data):
    # d = data.deepcopy()
    d = copy.deepcopy(data)
    features = d.columns
    size = int(np.floor(np.sqrt(features.size)))
    return np.random.choice(features, size=size, replace=False)

def main():
    data = pd.read_csv('./mutations.csv', index_col='Unnamed: 0')
    trees = []    
    out_of_bags = []

    for i in range(NUMBER_OF_TREES):
        # grab sqrt(n) columns, 230 columns with replacement
        samples = get_samples(data.index)
        mutations = get_features(data)
        data_set = data.loc[samples, mutations]
        out_of_bag = data.drop(index=samples, columns=mutations)
        decision_tree = DecisionTreePhi(data_set, 2)
        trees.append(decision_tree)
        out_of_bags.append(out_of_bag)

        print('iteration: ', i+1, '====================')
        print('Out of Bag size: ', len(out_of_bag))
        print('Out of Bag: ', out_of_bag)
        print('Tree: ')
        print(decision_tree)

    print('================================')
    print('Avergage Out of Bag size: ', np.average(len(out_of_bags)))
    
        


if __name__ == '__main__':
    main()
