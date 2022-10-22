import pandas as pd
import decisiontree as dt

data = pd.read_csv('./mutations.csv', index_col='Unnamed: 0')

decision_tree = dt.DecisionTreePhi(data=data, depth=2)
print(decision_tree)



# def get_next_classifier(self, matrices):
#     """find the best mutation to use to classify cancer based on true and false 
#         positives"""
    
#     frame_columns=['n(t_l)', 'n(t_r)', 'n(t_l, C)', 'n(t_l, NC)', 'P_l', 'P_r', 'P(C | t_l)', \
#                    'P(NC | t_l)', 'P(C | t_r)', 'P(NC | t_r)', '2*P_l*P_r', 'Q', 'Phi(s, t)']
    
#     selection_frame = pd.DataFrame(columns=frame_columns)
    
#     classified_positive = self.data.apply(lambda x: x[x==1].index, axis=0)
#     classified_negative = self.data.apply(lambda x: x[x==0].index, axis=0)
    
#     # get the number of samples in each tree
#     selection_frame['n(t_r)'] = classified_positive.apply(lambda x: x.size)
#     selection_frame['n(t_l)'] = classified_negative.apply(lambda x: x.size)
    
#     # get the number of cancer and non-cancer in the left tree (fp and tn in this tree)
#     selection_frame['n(t_l, C)'] = classified_negative.apply(lambda x: x[x.str.startswith('C')].size)
#     selection_frame['n(t_l, NC)'] = classified_negative.apply(lambda x: x[x.str.startswith('NC')].size)
    
#     # get the probability of being in the left or right tree at this split
#     selection_frame['P_l'] = selection_frame.apply(lambda x: x['n(t_l)'] / self.data.shape[0], axis=1)
#     selection_frame['P_r'] = selection_frame.apply(lambda x: x['n(t_r)'] / self.data.shape[0], axis=1)
    
#     # probability of having cancer and non-cancer in the left tree 
#     selection_frame['P(C | t_l)'] = selection_frame.apply(lambda x: x['n(t_l, C)'] / x['n(t_l)'], axis=1)
#     selection_frame['P(NC | t_l)'] = selection_frame.apply(lambda x: x['n(t_l, NC)'] / x['n(t_l)'], axis=1)
    
#     # probability of having cancer and non-cancer in the right tree
#     selection_frame['P(C | t_r)'] = selection_frame.apply(lambda x: (1 - x['n(t_l, C)']) / x['n(t_r)'], axis=1)
#     selection_frame['P(NC | t_r)'] = selection_frame.apply(lambda x: (1-x['n(t_l, NC)']) / x['n(t_r)'], axis=1)
    
#     # balance
#     selection_frame['2*P_l*P_r'] = selection_frame.apply(lambda x: 2 * x['P_l'] * x['P_r'], axis=1)
    
#     # purity
#     selection_frame['Q'] = selection_frame.apply(lambda x: abs(x['P(C | t_l)'] - x['P(C | t_r)']) + abs(x['P(NC | t_l)'] - x['P(NC | t_r)']), axis = 1)
    
#     selection_frame['Phi(s, t)'] = selection_frame.apply(lambda x: x['2*P_l*P_r'] * x['Q'], axis=1)
    
#     # frame = frame.sort_values(by='Phi(s, t)', ascending=False)
#     # print(frame.head(n=10))
    
#     selection_frame = selection_frame.sort_values(by='Phi(s, t)', ascending=False)
#     print(selection_frame.head(n=10))
    
    
    








        # frame = pd.DataFrame(columns=frame_columns)

        # for i in self.data.columns:
        #     col = self.data.loc[:, i]

            

        #     # do the work for the right sub tree (positives)
        #     # num_samples_classified_positive = col.aggregate('sum')
        #     samples_classified_positive = col[col == 1].index.to_list()
        #     num_samples_classified_positive = len(samples_classified_positive)
        #     num_true_positives = len(list(map(lambda x: x.startswith('C'), \
        #                                   samples_classified_positive)))
        #     num_false_positives = num_samples_classified_positive - num_true_positives
        #     prob_cancer_in_positives = round(num_true_positives / \
        #                                      num_samples_classified_positive, 3)
        #     prob_non_cancer_in_positives = round(num_false_positives / \
        #                                          num_samples_classified_positive, 3)
        #     prob_in_positive = num_samples_classified_positive / col.size

        #     # do the work for the left sub tree (negatives)
        #     num_samples_classified_negative = col.size - num_samples_classified_positive
        #     samples_classified_negative = col[col == 0].index.to_list()
        #     num_false_negatives = len(list(map(lambda x: x.startswith('C'),\
        #                                        samples_classified_negative)))
        #     num_true_negatives = num_samples_classified_negative - num_false_negatives
        #     prob_cancer_in_negatives = round(num_false_negatives / \
        #                                      num_samples_classified_positive, 3)
        #     prob_non_cancer_in_negatives = round(num_true_negatives / \
        #                                          num_samples_classified_positive, 3)
        #     prob_in_negative = num_samples_classified_negative / col.size

        #     # how uniform is the split at this mutation
        #     q = abs(prob_cancer_in_negatives - prob_cancer_in_positives) + \
        #         abs(prob_non_cancer_in_negatives - prob_non_cancer_in_positives)

        #     # how balance in size are the splits at this mutation
        #     balance = round(2 * prob_in_negative * prob_in_positive, 3)

        #     # combine q with how balanced in size the splits are
        #     phi = round(balance * q, 3)

        #     series_data = {'n(t_l)': num_samples_classified_negative,\
        #                    'n(t_r)': num_samples_classified_positive,\
        #                    'n(t_l, C)': num_false_positives,\
        #                    'P_l': prob_in_negative,\
        #                    'P_r': prob_in_positive,\
        #                    'P(C | t_l)': prob_cancer_in_negatives,
        #                    'P(NC | t_l)': prob_non_cancer_in_negatives,\
        #                    'P(C | t_r)': prob_cancer_in_positives,\
        #                    'P(NC | t_r)': prob_non_cancer_in_positives,\
        #                    '2*P_l*P_r': balance,\
        #                    'Q': q,\
        #                    'Phi(s, t)': phi,
        #     }
        #     series = pd.Series(data=series_data, name=i)
        #     frame = frame.append(series)

        # frame = frame.sort_values(by='Phi(s, t)', ascending=False)
        # print(frame.head(n=10))
        # return frame.iloc[0,:].name
