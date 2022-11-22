import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# TODO: we don't use self.tree[i]['positive_data'] (and negative data)
#         can we get rid of it, or use it?

class BinaryTreeArray:
    def __init__(self, data, depth):
        self.data = data
        self.depth = depth
        self.leaf_start = (2**(self.depth-1))-1
        temp_object = {'data': None, 'name': 'leaf', 'classification': None, \
                       'positive_data': None, 'negative_data': None}
        self.tree = [temp_object for i in range((2**depth)-1)]

    def __str__(self):
        names = []
        for i in range(len(self.tree)):
            names.append(self.tree[i]['name'])
        return str(names)

class DecisionTreePhi(BinaryTreeArray):
    def __init__(self, data, depth):
        super().__init__(data, depth)

        for i in range(len(self.tree)):
            if i == 0:
                self.tree[i]['data'] = self.data

            # not in the leaf nodes yet, or in a single node tree
            if i < self.leaf_start:
                (name, positive, negative) = \
                    self.__get_next_classifier(self.tree[i]['data'], i)
                self.tree[i]['name'] = name

            if self.depth > 1 and i < self.leaf_start:
                self.tree[(2*i)+1]['data'] = data.drop(positive)
                self.tree[(2*i)+2]['data'] = data.drop(negative)

            # leaf nodes
            if i >= self.leaf_start:
                # determine if we are cancer node or not
                print('==========')
                print(i)
                self.tree[i]['classification'] = \
                    self.__derive_classification(self.tree[i]['data'])

    def __derive_classification(self, data):
        num_cancer = data[data.index.str.startswith('C')].shape[0]
        num_non_cancer = data[data.index.str.startswith('NC')].shape[0]
        print('C', num_cancer)
        print('NC:', num_non_cancer)
        print(data[data.index.str.startswith('C')].shape[0])
        print(data[data.index.str.startswith('NC')].shape[0])
        return num_cancer >= num_non_cancer

    def classify(self, sample):
        """Determine if a given input sample has cancer or not according to our tree"""
        i = 0
        while i < self.leaf_start:
            if sample[self.tree[i]['name']]:
                # has mutation, go to right child
                i = (2 * i) + 2
            else:
                # lacks mutation, go to left child
                i = (2 * i) + 1
        return self.tree[i]['classification']

    def __get_next_classifier(self, data, tree_index):
        frame_columns = ['n(t_l)', 'n(t_r)', 'n(t_l, C)', 'n(t_l, NC)', 'n(t_r, C)', \
                         'n(t_r, NC)', 'P_l', 'P_r', 'P(C | t_l)', 'P(NC | t_l)', \
                         'P(C | t_r)', 'P(NC | t_r)', '2*P_l*P_r', 'Q', 'Phi(s, t)']

        selection_frame = pd.DataFrame(columns=frame_columns)
        classified_positive = self.data.apply(lambda x: x[x==1].index, axis=0)
        classified_negative = self.data.apply(lambda x: x[x==0].index, axis=0)
        
        # get the number of samples in each tree
        selection_frame['n(t_r)'] = \
            classified_positive.apply(lambda x: 1 if (x.size < 1) else x.size)
        selection_frame['n(t_l)'] = \
            classified_negative.apply(lambda x: 1 if (x.size < 1) else x.size)

        # get the number of cancer and non-cancer in the left tree (fp and tn in this tree)
        selection_frame['n(t_l, C)'] = \
            classified_negative.apply(lambda x: x[x.str.startswith('C')].size)
        selection_frame['n(t_l, NC)'] = \
            classified_negative.apply(lambda x: x[x.str.startswith('NC')].size)

        # get the number of cancer and non-cancer in the left tree (fp and tn in this tree)
        selection_frame['n(t_r, C)'] = \
            classified_positive.apply(lambda x: x[x.str.startswith('C')].size)
        selection_frame['n(t_r, NC)'] = \
            classified_positive.apply(lambda x: x[x.str.startswith('NC')].size)

        # get the probability of being in the left or right tree at this split
        selection_frame['P_l'] = \
            selection_frame.apply(lambda x: x['n(t_l)'] / self.data.shape[0], axis=1)
        selection_frame['P_r'] = \
            selection_frame.apply(lambda x: x['n(t_r)'] / self.data.shape[0], axis=1)
        
        # probability of having cancer and non-cancer in the left tree 
        selection_frame['P(C | t_l)'] = \
            selection_frame.apply(lambda x: x['n(t_l, C)'] / x['n(t_l)'], axis=1)
        selection_frame['P(NC | t_l)'] = \
            selection_frame.apply(lambda x: x['n(t_l, NC)'] / x['n(t_l)'], axis=1)
        
        # probability of having cancer and non-cancer in the right tree
        selection_frame['P(C | t_r)'] = \
            selection_frame.apply(lambda x: x['n(t_r, C)'] / x['n(t_r)'], axis=1)
        selection_frame['P(NC | t_r)'] = \
            selection_frame.apply(lambda x: x['n(t_r, NC)'] / x['n(t_r)'], axis=1)
        
        # balance
        selection_frame['2*P_l*P_r'] = \
            selection_frame.apply(lambda x: 2 * x['P_l'] * x['P_r'], axis=1)
        
        # purity
        selection_frame['Q'] = \
            selection_frame.apply(lambda x: abs(x['P(C | t_l)'] - \
                                                x['P(C | t_r)']) + \
                                  abs(x['P(NC | t_l)'] - x['P(NC | t_r)']), axis = 1)
        
        selection_frame['Phi(s, t)'] = \
            selection_frame.apply(lambda x: x['2*P_l*P_r'] * x['Q'], axis=1)
        
        selection_frame = selection_frame.sort_values(by='Phi(s, t)', ascending=False)

        name = selection_frame.iloc[0,:].name

        # if tree_index == 0:
        #     n_t = self.data.shape[0]
        #     n_tc = len(self.data[self.data.index.str.startswith('C')])
        #     n_tnc = len(self.data[self.data.index.str.startswith('NC')])
        #     print('n_t: ', n_t)
        #     print('n_tc: ', n_tc)
        #     print('n_tnc: ', n_tnc)
        #     print('p_ct: ', n_tc / n_t)
        #     print('p_nct: ', n_tnc / n_t)
        #     print(selection_frame.head(n=10))
        return name, classified_positive[name], classified_negative[name]
        
