import pandas as pd
import decisiontree as dt
import decisiontreetwo as dt2

data = pd.read_csv('./mutations.csv', index_col='Unnamed: 0')

def evaluate(confusion_matrix):
    print()
    print(confusion_matrix)
    print()
    # translate our confusion matrix from booleans to tp/fp/tn/fn
    for i in confusion_matrix:
        if i.startswith('NC'):
            confusion_matrix[i] = 'tn' if confusion_matrix[i] == False else 'fp'
        else:
            confusion_matrix[i] = 'tp' if confusion_matrix[i] == True else 'fn'
    print()
    print(confusion_matrix)
    print()            
            
    # after classifying everything, we need to determine:
    # accuracy, sensitivity, specificity, precision, miss rate,
    # false discovery rate, and false omission rate
    # we also need to plot a confusion matrix
        
    classified_tp_sum = 0
    classified_tn_sum = 0
    classified_fp_sum = 0
    classified_fn_sum = 0

    tp = {}
    tn = {}
    fp = {}
    fn = {}

    for i in confusion_matrix:
        if confusion_matrix[i] == 'tp':
            classified_tp_sum += 1
        elif confusion_matrix[i] == 'tn':
            classified_tn_sum += 1
        elif confusion_matrix[i] == 'fp':
            classified_fp_sum += 1
        else:
            classified_fn_sum += 1

    print('LENGTH:',len(confusion_matrix))
    accuracy = (classified_tp_sum + classified_tn_sum) / len(confusion_matrix)
    # sensitivity = classified_tp_sum / (classified_tp_sum + classified_fn_sum)
    # specificity = classified_tn_sum / (classified_tn_sum + classified_fp_sum)
    # precision = classified_tp_sum / (classified_tp_sum + classified_fp_sum)
    # miss_rate = 1 - sensitivity
    # fdr = 1 - precision
    # false_omission_rate = classified_fn_sum / (classified_fn_sum + classified_tn_sum)
    # return accuracy, sensitivity, specificity, precision, miss_rate, fdr, false_omission_rate
    return accuracy


decision_tree = dt.DecisionTreePhi(data=data, depth=3)
# decision_tree = dt.DecisionTreeGain(data = data, depth = 3)
# decision_tree = dt.DecisionTree(data, 2)

# decision_tree = dt2.DecisionTreePhi(data=data, depth=3)

confusion_matrix = {}

for k in data.index:
    classification_phi = decision_tree.classify(data.loc[k, :])
    confusion_matrix[k] = classification_phi

# (accuracy, sensitivity, specificity, precision, miss_rate, fdr, false_omission_rate) = evaluate(confusion_matrix)
accuracy = evaluate(confusion_matrix)

print(decision_tree)
print('==================================================')
print(accuracy)
