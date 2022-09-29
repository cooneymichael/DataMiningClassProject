import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DecisionTree:
    # TODO: replace function parameters with self contained variables
    def __init__(self, data, depth, classifier=None, remove_used=False):
        self.data = data
        self.classifier = classifier
        self.depth = depth

        # root node setup:
        if self.classifier == None:
            # find the best classifier
            matrices = self.__generate_confusion_matrices(self.data)
            self.classifier = self.__get_next_classifier(matrices)

        # segregate the data using the classifier
        positives, negatives = self.__segregate_data(self.data[self.classifier])

        # generate confusion matrices with the segrated data
        # TODO: get "data" into the tree
        positive_confusion_matrix_data = self.__generate_confusion_matrices(data.loc[positives, :])
        negative_confusion_matrix_data = self.__generate_confusion_matrices(data.loc[negatives, :])

        # find the next classifier
        positive_classifier = self.__get_next_classifier(positive_confusion_matrix_data)
        negative_classifier = self.__get_next_classifier(negative_confusion_matrix_data)

        if depth > 1:
            self.right = DecisionTree(self.data, self.depth - 1, classifier=positive_classifier)
            self.left = DecisionTree(self.data, self.depth-1, classifier=negative_classifier)
        else:
            self.right = None
            self.Left = None

    # pseudo decision tree (I would like to do this with an actual binary tree)
    def __segregate_data(self, mutation_of_interest):
        """Discern whether a sample tests as positive or negative given a mutation"""
        positive = []
        negative = []
        samples = mutation_of_interest.index

        for idx, val in enumerate(mutation_of_interest.values):
            positive.append(samples[idx]) if (val == 1) else negative.append(samples[idx])

        # print()
        # print("========== Positives ==========")
        # print(positive)
        # print(len(positive))

        # print()
        # print("========== Negatives ==========")
        # print(negative)
        # print(len(negative))

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
        # sorted_diffs = sorted(statistics, key=lambda x: statistics[x][0], reverse=True)[:10]
        # sorted_percents = sorted(statistics, key=lambda x: statistics[x][0], reverse=True)[:10]
        # return (sorted_diffs, sorted_percents)

    def classify(self, sample):
        # TODO: figure out why classifiers are stored in a list -> noticed while
        # writing tree and too scared to fix it right now
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

        # print(self.classifier)
        # print("{ ")

        #     print(self.left)
        #     print("} {")
        #     print(self.right)
        # else:
        #     return ("{Non Cancer}{Cancer}")
        #     # print("Non Cancer")
        #     # print("} {")
        #     # print("Cancer")
        # print("}")


# plan:
# root node has depth
# uses given mutation to classify data
# classifies subsets of data, finding next two classifiers
# creates child nodes with depth-1 -> repeat the process
# if depth = 1: stop

# methods:
#   segregate_data:
#     given a mutation: do the training samples have cancer or not
#
#   generate_confusion_matrix
#
#   classify:
#     take in a sample and pass it recursively through the tree:
#     If it has the mutation in the current node:
#       If depth is not 1:
#         Call classify on right node with sample
#       Else:
#         Return Cancer
#     If it does not have the mutation in the current node:
#       If depth is not 1:
#         Call classify on the left node with sample
#       Else:
#         Return Non-Cancer
#     
#
#   
