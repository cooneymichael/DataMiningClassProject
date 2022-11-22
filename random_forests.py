import pandas as pd
import numpy as np
from decisiontree import *
import copy
# from decisiontreetwo import DecisionTreePhi

NUMBER_OF_TREES = 25

def get_samples(index):
    return np.random.choice(index, size=index.size)

def get_features(data):
    # d = data.deepcopy()
    d = copy.deepcopy(data)
    features = d.columns
    size = int(np.floor(np.sqrt(features.size)))
    return np.random.choice(features, size=size, replace=False)


def forest_classifier(sample, trees):
    votes = [False for i in range(len(trees))]
    for i in range(len(trees)):
        votes[i] = trees[i].classify(sample)
    return (votes.count(True) > votes.count(False), votes.count(True), votes.count(False))


def evaluate(confusion_matrix):
    # translate our confusion matrix from booleans to tp/fp/tn/fn
    for i in confusion_matrix:
        if i.startswith('NC'):
            confusion_matrix[i] = 'tn' if confusion_matrix[i] == False else 'fp'
        else:
            confusion_matrix[i] = 'tp' if confusion_matrix[i] == True else 'fn'
            
            
    # after classifying everything, we need to determine:
    # accuracy, sensitivity, specificity, precision, miss rate,
    # false discovery rate, and false omission rate
    # we also need to plot a confusion matrix
        
    classified_tp_sum = 0
    classified_tn_sum = 0
    classified_fp_sum = 0
    classified_fn_sum = 0
    for i in confusion_matrix:
        if confusion_matrix[i] == 'tp':
            classified_tp_sum += 1
        elif confusion_matrix[i] == 'tn':
            classified_tn_sum += 1
        elif confusion_matrix[i] == 'fp':
            classified_fp_sum += 1
        else:
            classified_fn_sum += 1

    accuracy = (classified_tp_sum + classified_tn_sum) / len(confusion_matrix)
    denom_sens = 1 if (classified_tp_sum + classified_fn_sum == 0) else (classified_tp_sum + classified_fn_sum)
    sensitivity = classified_tp_sum / denom_sens
    denom_spec = 1 if (classified_tn_sum + classified_fp_sum == 0) else (classified_tn_sum + classified_fp_sum)
    specificity = classified_tn_sum / denom_spec
    denom_prec = 1 if (classified_tp_sum + classified_fp_sum == 0) else (classified_tp_sum + classified_fp_sum)
    precision = classified_tp_sum / denom_prec
    miss_rate = 1 - sensitivity
    fdr = 1 - precision
    denom = 1 if (classified_fn_sum + classified_tn_sum == 0) else (classified_fn_sum + classified_tn_sum)
    false_omission_rate = classified_fn_sum / denom
    return accuracy, sensitivity, specificity, precision, miss_rate, fdr, false_omission_rate, \
        classified_tp_sum, classified_tn_sum, classified_fp_sum, classified_fn_sum



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

    print('================================')
    print('Classifying Samples')

    samples_list = ['C1', 'C10', 'C50', 'NC5', 'NC15']
    for i in samples_list:
        print()
        sample_data = data.loc[i,:]
        classification, votes_positive, votes_negative = forest_classifier(sample_data, trees)
        print('Classification Results for', i)
        print('Classification:', classification, '\tVotes Positive:', votes_positive, \
              '\tVotes Negative:', votes_negative)

    print()
    print('================================')
    print('Average size of the Out of Bag set')
    out_of_bag_sum = 0
    for i in out_of_bags:
        out_of_bag_sum += len(i)

    out_of_bag_sum /= len(out_of_bags)
    print(out_of_bag_sum)

    print()
    print('================================')
    print('Classifying Out of Bag Set-0')
    print()
    confusion_matrix = {}

    for i in out_of_bags[0].index:
        sample_data = data.loc[i,:]
        classification, votes_positive, votes_negative = forest_classifier(sample_data, trees)
        # if classification:
        #     print('TRUE!!!!!!!!!!!!!!!!!!!')
        confusion_matrix[i] = classification
    
    accuracy, sensitivity, specificity, precision, miss_rate, fdr, false_omission_rate, tp, tn, fp, fn\
        = evaluate(confusion_matrix)
    print('Accuracy:', accuracy)
    print('Sensitivity:', sensitivity)
    print('Specificity:', specificity)
    print('Precision:', precision)
    print('Miss Rate:', miss_rate)
    print('FDR:', fdr)
    print('FOR:', false_omission_rate)

    print('tp:', tp)
    print('tn:', tn)
    print('fp:', fp)
    print('fn:', fn)
    


if __name__ == '__main__':
    main()
