import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log

import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)

# TODO: remove dead code, pseudo code
# TODO: find a good way to prevent already-used mutations from being new classifiers
class DecisionTree:
    def __init__(self, data, depth, classifier=None, remove_used=False):
        self.data = data
        self.classifier = classifier
        self.depth = depth
        self.positives = None
        self.negatives = None

        # root node setup:
        if self.classifier == None:
            # find the best classifier
            matrices = self.__generate_confusion_matrices(self.data)
            self.classifier = self.__get_next_classifier(matrices)

        # segregate the data using the classifier
        positives, negatives = self.__segregate_data(self.data[self.classifier])

        if depth > 1:
            # generate confusion matrices with the segrated data
            # TODO: get "data" into the tree
            positive_confusion_matrix_data = \
                self.__generate_confusion_matrices(data.loc[positives, :])
            negative_confusion_matrix_data = \
                self.__generate_confusion_matrices(data.loc[negatives, :])
            
            # find the next classifier
            positive_classifier = \
                self.__get_next_classifier(positive_confusion_matrix_data)
            negative_classifier = \
                self.__get_next_classifier(negative_confusion_matrix_data)

        if depth > 1:
            positive_data = self.data.drop(negatives, axis=0)
            negative_data = self.data.drop(positives, axis=0)
            self.right = \
                DecisionTree(positive_data, self.depth - 1, classifier=positive_classifier)
            self.left = \
                DecisionTree(negative_data, self.depth-1, classifier=negative_classifier)
        else:
            self.right = None
            self.Left = None

    def __segregate_data(self, mutation_of_interest):
        """Discern whether a sample tests as positive or negative given a mutation"""
        positive = []
        negative = []
        samples = mutation_of_interest.index

        for idx, val in enumerate(mutation_of_interest.values):
            positive.append(samples[idx]) if (val == 1) else negative.append(samples[idx])

        return positive, negative

    def __generate_confusion_matrices(self, data):
        """Iterates through each mutation in the population and determines how many 
        true and false postives, and true and false negatives there are"""
        mutations = data.columns
        binary_labels = list(map(lambda x: 1 if x.startswith('C') else 0, data.index))
        matrices = {}
        for i in range(len(mutations)):
            col = data[mutations[i]]
            final_data = data[mutations[i]]
            df = pd.DataFrame(columns=[mutations[i]], index=data.index)
            
            final_data = list(map(lambda x, y: 'tp' if (x and y) else ('fp' if y == 1 else y), binary_labels, col))
            final_data = list(map(lambda x, y: 'tn' if (not x and  not y) else ('fn' if y == 0 else y), binary_labels, final_data))
            df[mutations[i]] = final_data
            
            matrices[mutations[i]] = df
        return matrices

    def __get_next_classifier(self, matrices):
        """find the best mutation to use to classify cancer based on true and false 
        positives"""
        
        # Pseudocode:
        # for i in matrices:
        #     # calculate TP-FP and %TP-%FP
        #     Sum tp and fp, take difference
        #     find % of tp and fp, take difference
        #     store in a tuple in a dict, e.g.: {'RNF...': (diff, %diff)}
        # sort list by max value of each and print out the data to the screen

        statistics = {}
        for i in matrices:
            df = matrices[i]
            tp = len(df[df.values == 'tp'])
            fp = len(df[df.values == 'fp'])
            
            percent_tp = round(tp / 230, 2)
            percent_fp = round(fp / 230, 2)
            statistics[i] = (tp - fp, round(percent_tp - percent_fp, 2))
        return sorted(statistics, key=lambda x: statistics[x][0], reverse=True)[:1]

    def classify(self, sample):
        if sample[self.classifier[0]]:
            if self.depth > 1:
                # TODO: finish this
                return self.right.classify(sample)
            else:
                return True
        else:
            if self.depth > 1:
                return self.left.classify(sample)
            else:
                return False

    def plot_confusion_matrix(self, plot_depth=1):
        if plot_depth > 1:
            self.left.plot_confusion_matrix(plot_depth - 1)
            self.right.plot_confusion_matrix(plot_depth - 1)
            return
        else:

            fig, ax = plt.subplots(ncols=1, nrows=1)

            # get 'data' that is just calssifier and index
            ind_data = pd.DataFrame(index=self.data.index, data=self.data[self.classifier[0]])
            matrices = self.__generate_confusion_matrices(ind_data)
            confusion_matrix = matrices[self.classifier[0]]
            
            tp = len(confusion_matrix[confusion_matrix.values == 'tp'])
            tn = len(confusion_matrix[confusion_matrix.values == 'tn'])
            fp = len(confusion_matrix[confusion_matrix.values == 'fp'])
            fn = len(confusion_matrix[confusion_matrix.values == 'fn'])

            # generate 4 colored rectangles, add labels to them
            ax.imshow([[0.0, 0.7], [1.3, 2]], interpolation='nearest', cmap='PiYG')
            ax.set_xticks(np.arange(0,2), ['positive', 'negative'])
            ax.set_yticks(np.arange(0,2), ['positive', 'negative'])
            ax.set_xlabel('PREDICTED')
            ax.set_ylabel('ACTUAL')
            ax.text(0,0,'TP\n' + str(tp))
            ax.text(0,1,'FP\n' + str(fp))
            ax.text(1,0,'FN\n' + str(fn))
            ax.text(1,1,'TN\n' + str(tn))
            ax.set_title(confusion_matrix.columns.values[0])


    def __str__(self):
        return '{ ' + self.classifier[0]\
            + '\n{ ' + (str(self.left) if self.depth > 1 else 'NC') + ' }'\
            + '\n{ ' + (str(self.right) if self.depth > 1 else 'C') + ' }'\
            + '\n}'




################################################################################
################################################################################
################################################################################



class DecisionTreePhi:
    def __init__(self, data, depth, classifier=None, remove_used=False, root=True):
        self.data = data
        self.classifier = classifier
        self.depth = depth
        self.root = root
        self.positives = None
        self.negatives = None
        self.class_negative_cancerous = None
        self.class_positive_cancerous = None
        
        # root node setup:
        if self.classifier == None:
            # find the best classifier
            # matrices = self.__generate_confusion_matrices(self.data)
            # self.classifier = self.__get_next_classifier(matrices)
            self.classifier = self.__get_next_classifier()
            
        # segregate the data using the classifier
        # self.positives, self.negatives = self.__segregate_data(self.data[self.classifier])

        if depth > 1:
            positive_data = self.data.drop(self.positives.values, axis=0)
            negative_data = self.data.drop(self.negatives.values, axis=0)
            self.right = DecisionTreePhi(positive_data, self.depth - 1, root=False)
            self.left = DecisionTreePhi(negative_data, self.depth-1, root=False)
        else:
            self.right = None
            self.Left = None

            # determine the classification schema if we are in the leaf nodes
            num_pos = self.data[self.data.index.str.startswith('C')].shape[0]
            num_neg = self.data[self.data.index.str.startswith('NC')].shape[0]
            self.class_negative_cancerous = num_neg > num_pos
            self.class_positive_cancerous = num_pos >= num_neg


    def classify(self, sample):
        if sample[self.classifier]:
            if self.depth > 1:
                # TODO: finish this
                return self.right.classify(sample)
            else:
                return self.class_positive_cancerous
        else:
            if self.depth > 1:
                return self.left.classify(sample)
            else:
                return self.class_negative_cancerous


    def __get_next_classifier(self):
        """find the best mutation to use to classify cancer based on true and false 
        positives"""
        
        
        frame_columns=['n(t_l)', 'n(t_r)', 'n(t_l, C)', 'n(t_l, NC)', 'n(t_r, C)', 'n(t_r, NC)', 'P_l', 'P_r', 'P(C | t_l)', \
                       'P(NC | t_l)', 'P(C | t_r)', 'P(NC | t_r)', '2*P_l*P_r', 'Q', 'Phi(s, t)']
        
        selection_frame = pd.DataFrame(columns=frame_columns)
        
        classified_positive = self.data.apply(lambda x: x[x==1].index, axis=0)
        classified_negative = self.data.apply(lambda x: x[x==0].index, axis=0)
        
        # get the number of samples in each tree
        selection_frame['n(t_r)'] = \
            classified_positive.apply(lambda x: 1 if (x.size < 1) else x.size)
        selection_frame['n(t_l)'] = \
            classified_negative.apply(lambda x: 1 if (x.size < 1) else x.size)

        # selection_frame['n(t_r)'] = selection_frame.apply(lambda x: 1 if x.loc['n(t_r)'] == 0 else x.loc['n(t_r)'], axis=1)
        # selection_frame['n(t_r)'] = selection_frame.apply(lambda x: 1 if x.loc['n(t_l)'] == 0 else x.loc['n(t_l)'], axis=1)

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

        # if self.root:
        #     n_t = self.data.shape[0]
        #     n_tc = len(self.data[self.data.index.str.startswith('C')])
        #     n_tnc = len(self.data[self.data.index.str.startswith('NC')])
        #     print('n_t: ', n_t)
        #     print('n_tc: ', n_tc)
        #     print('n_tnc: ', n_tnc)
        #     print('p_ct: ', n_tc / n_t)
        #     print('p_nct: ', n_tnc / n_t)
        #     print(selection_frame.head(n=10))
            


        self.positives = classified_positive[name]
        self.negatives = classified_negative[name]

        # self.class_negative_cancerous = \
        #     selection_frame.loc[name, 'n(t_l, C)'] > \
        #     selection_frame.loc[name, 'n(t_l, NC)']

        # self.class_positive_cancerous = \
        #     selection_frame.loc[name, 'n(t_r, C)'] > \
        #     selection_frame.loc[name, 'n(t_r, NC)']

        return name


    def __str__(self):
        return '{ ' + self.classifier\
            + '\n{ ' + (str(self.left) if self.depth > 1 else 'NC') + ' }'\
            + '\n{ ' + (str(self.right) if self.depth > 1 else 'C') + ' }'\
            + '\n}'



################################################################################
################################################################################
################################################################################


class DecisionTreeGain:
    def __init__(self, data, depth):
        self.data = data
        self.depth = depth
        self.tree = [0 for i in range((2**depth)-1)]
        self.leaf_start = (2**(self.depth-1))-1
        # pseudo code/requirements:
        # each node will contain:
        #  name: feature name
        #  classification: True or False
        #  positive_data: elements of data classified as positive
        #  negative_data: elements of data classified as negative

        for i in range(len(self.tree)):
            self.tree[i] = {'data': None, 'name': 'leaf', 'classification': None, \
                            'positive_data': None, 'negative_data': None}

        for i in range(len(self.tree)):
            if i == 0:
                self.tree[i]['data'] = self.data

            # not in leaf nodes yet, or in single node tree
            if i < self.leaf_start:
                (name, positive, negative) = \
                    self.__get_next_classifier(self.tree[i]['data'], i)
                self.tree[i]['name'] = name

            # not in leaf nodes yet, and not a single node tree
            if self.depth > 1 and i < self.leaf_start:
                # left child, classified negative
                self.tree[(2*i)+1]['data'] = data.drop(positive)
                # right child, classified positive
                self.tree[(2*i)+2]['data'] = data.drop(negative)

            # leaf nodes
            if i >= self.leaf_start:
                # determine if we are a cancer node or not
                self.tree[i]['classification'] = \
                    self.__derive_classification(self.tree[i]['data'])

    def __derive_classification(self, data):
        num_cancer = data[data.index.str.startswith('C')].shape[0]
        num_non_cancer = data[data.index.str.startswith('NC')].shape[0]
        return num_cancer >= num_non_cancer

    def __get_next_classifier(self, data, tree_index):
        # create selection_frame
        # for each feature:
        #  calculate number of samples in left and right
        #  
        # find the top one, return name
        frame_columns=['n(t)', 'n(t, C)', 'n(t, NC)', 'n(t_l)', 'n(t_r)', 'n(t_l, C)', \
                       'n(t_l, NC)', 'n(t_r, C)', 'n(t_r, NC)', 'P_l', 'P_r', 'H(s, t)',\
                       'H(t_l)', 'H(t_r)', 'H(t)', 'gain(s)']
        
        selection_frame = pd.DataFrame(columns=frame_columns)

        # get the number of samples in this node
        selection_frame['n(t)'] = data.apply(lambda x: data.shape[0])

        # get the number of cancer and non-cancer in this node
        num_cancer = data[data.index.str.startswith('C')].shape[0]
        num_non_cancer = data[data.index.str.startswith('NC')].shape[0]
        selection_frame['n(t, C)'] = selection_frame.apply(lambda x: num_cancer, axis=1)
        selection_frame['n(t, NC)'] = selection_frame.apply(lambda x: num_non_cancer, \
                                                            axis=1)

        classified_positive = data.apply(lambda x: x[x==1].index, axis=0)
        classified_negative = data.apply(lambda x: x[x==0].index, axis=0)

        # get the number of samples in each tree
        selection_frame['n(t_r)'] = classified_positive.apply(lambda x: 1 if (x.size < 1) \
                                                              else x.size)
        selection_frame['n(t_l)'] = classified_negative.apply(lambda x: 1 if (x.size < 1) \
                                                              else x.size)

        # get the number of cancer and non-cancer in the left tree (fp and tn in this tree)
        selection_frame['n(t_l, C)'] = \
            classified_negative.apply(lambda x: 1 if (x[x.str.startswith('C')].size < 1) \
                                      else x[x.str.startswith('C')].size)
        selection_frame['n(t_l, NC)'] = \
            classified_negative.apply(lambda x: 1 if (x[x.str.startswith('NC')].size < 1) \
                                      else x[x.str.startswith('NC')].size)

        # get the number of cancer and non-cancer in the left tree (fp and tn in this tree)
        selection_frame['n(t_r, C)'] = \
            classified_positive.apply(lambda x: 1 if (x[x.str.startswith('C')].size < 1) \
                                      else x[x.str.startswith('C')].size)
        selection_frame['n(t_r, NC)'] = \
            classified_positive.apply(lambda x: 1 if (x[x.str.startswith('NC')].size < 1) \
                                      else x[x.str.startswith('NC')].size)

        # get the probability of being in the left or right tree at this split
        selection_frame['P_l'] = \
            selection_frame.apply(lambda x: x['n(t_l)'] / self.data.shape[0], axis=1)
        selection_frame['P_r'] = \
            selection_frame.apply(lambda x: x['n(t_r)'] / self.data.shape[0], axis=1)

        # find H(t_l)
        selection_frame['H(t_l)'] = \
            selection_frame.apply(lambda x: -1 \
                                  * ((x['n(t_l, C)'] / x['n(t_l)'] \
                                      * log(x['n(t_l, C)'] / x['n(t_l)'], 2))\
                                     + (x['n(t_l, NC)'] / x['n(t_l)'] \
                                        * log(x['n(t_l, NC)'] / x['n(t_l)'], 2))), axis=1)

        # find H(t_r) (not an accurate name)
        selection_frame['H(t_r)'] = \
            selection_frame.apply(lambda x: -1 \
                                  * ((x['n(t_r, C)'] / x['n(t_r)'] \
                                      * log(x['n(t_r, C)'] / x['n(t_r)'], 2))\
                                     + (x['n(t_r, NC)'] / x['n(t_r)'] \
                                        * log(x['n(t_r, NC)'] / x['n(t_r)'], 2))), axis=1)


        # find H(t,s), the entropy of this split at this feature
        selection_frame['H(s, t)'] = \
            selection_frame.apply(lambda x: (x['P_l'] * x['H(t_l)']) \
                                  + (x['P_r'] * x['H(t_r)']), axis=1)

        # find H(t), the entropy at this node
        selection_frame['H(t)'] = \
            selection_frame.apply(lambda x: -1 \
                                  * ((x['n(t, C)'] / x['n(t)'] \
                                      * log(x['n(t, C)'] / x['n(t)'], 2))\
                                     +(x['n(t, NC)'] / x['n(t)'] \
                                       * log(x['n(t, NC)'] / x['n(t)'], 2))), axis=1)
        selection_frame['gain(s)'] = \
            selection_frame.apply(lambda x: x['H(t)'] - x['H(s, t)'], axis=1)

        selection_frame = selection_frame.sort_values(by='gain(s)', ascending=False)

        if tree_index == 0:
            print(selection_frame.head(n=10))

        name = selection_frame.iloc[0,:].name
        return name, classified_positive[name], classified_negative[name]

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

    def __str__(self):
        names = []
        for i in range(len(self.tree)):
            names.append(self.tree[i]['name'])
        return str(names)
        
