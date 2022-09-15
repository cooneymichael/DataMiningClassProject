import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as Annotation
import sys #for sys.argv
import getopt

# forgive me for my sins
number_of_mutations_per_sample = pd.DataFrame([])
number_of_samples_per_mutation = pd.DataFrame([])
samples = []
mutations = []

def on_pick(event):
    """Allow the user to click on a data point on the scatter plot and display its
    coordinates"""
    ind = event.ind
    if event.mouseevent.inaxes:
        ax = event.mouseevent.inaxes
        title = ax.get_title()
        if title == 'Number of mutations per sample':
            print('ax0: ', number_of_mutations_per_sample[ind], samples[ind])
        else:
            print('ax1: ', number_of_samples_per_mutation[ind], mutations[ind])

def plot_data(data, mut_per_sample, samples_per_mut, samples_local, muts):
    """Create two scatter plots, plot the data, and display it.  Scatter plots can
    be interacted with by the user and because of this will have no labels on the x-axes"""
    fig, ((ax0), (ax1)) = plt.subplots(nrows=2, ncols=1)
    plt.connect('pick_event', on_pick)

    # first scatter plot: number of mutations (y-axis) per sample (x-axis)
    # we remove the tick labels from the x-axis because it's too crowded to be legible,
    # but we can click on individual points and get the x and y values
    number_of_mutations_per_sample = mut_per_sample
    samples = samples_local
    ax0.scatter(samples, number_of_mutations_per_sample, marker='.', picker=5)
    ax0.axes.xaxis.set_ticklabels([])
    ax0.set_title('Number of mutations per sample')

    # second scatter plot: number of samples (y-axis) per mutation (x-axis)
    # the same logic applies to the omission of the x-axis labels here as above
    number_of_samples_per_mutation = samples_per_mut
    mutations = muts
    ax1.scatter(mutations, number_of_samples_per_mutation, marker='.', picker=5)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.set_title('Number of samples per mutation')


def get_top_ten(top_ten):
    """Sort each column of top_ten and print the ten highest values and their 
    corresponding mutations for each"""
    try:
        with open('topTenMutations.txt', 'w') as output:
            top_ten.sort_values('T', axis=0, inplace=True, ascending=False)
            output.write('Ten Highest Mutations by Total Number of Samples: ==============================\n')
            output.write(top_ten.head(n=10).to_string(header=True, index=True))
            output.write('\n')
            
            top_ten.sort_values('C', axis=0, inplace=True, ascending=False)
            output.write('Ten Highest Mutations by Total Cancer Samples: =================================\n')
            output.write(top_ten.head(n=10).to_string(header=True, index=True))
            output.write('\n')

            top_ten.sort_values('NC', axis=0, inplace=True, ascending=False)
            output.write('Ten Highest Mutations by Total Non Cancer Samples: =============================\n')
            output.write(top_ten.head(n=10).to_string(header=True, index=True))
            output.write('\n')

            top_ten.sort_values('%C', axis=0, inplace=True, ascending=False)
            output.write('Ten Highest Mutations by Percent Cancer Samples: ===============================\n')
            output.write(top_ten.head(n=10).to_string(header=True, index=True))
            output.write('\n')

            top_ten.sort_values('%NC', axis=0, inplace=True, ascending=False)
            output.write('Ten Highest Mutations by Percent Non Cancer Samples: ===========================\n')
            output.write(top_ten.head(n=10).to_string(header=True, index=True))
            output.write('\n')

            top_ten.sort_values('%C-%NC', axis=0, inplace=True, ascending=False)
            output.write('Ten Highest Mutations by the Difference between Cancer and Non Cancer Samples: =\n')
            output.write(top_ten.head(n=10).to_string(header=True, index=True))
            output.write('\n')

            top_ten.sort_values('%C/%NC', axis=0, inplace=True, ascending=False)
            output.write('Ten Highest Mutations by the Ratio of Cancer to Non Cancer Samples: ============\n')
            output.write(top_ten.head(n=10).to_string(header=True, index=True))
            output.write('\n')
    except OSError:
        print('OSError')
        
def explore_data(data, mut_per_sample, samples_per_mut, samples, muts):
    """Calculate some statistics about the mutations and cancer and non-cancer samples.
       The ten mutations with the highest statistical values for each statistic will be
       printed to a file"""
    number_of_mutations_per_sample = mut_per_sample
    number_of_samples_per_mutation = samples_per_mut
    samples = samples
    mutations = muts

    top_ten = pd.DataFrame(index=mutations, columns=['T', 'C', 'NC', '%C', '%NC', '%C-%NC', '%C/%NC'], data=0)
    top_ten['T'] = number_of_samples_per_mutation

    # for each column, which cells contain a 1 (True) and which contain a 0 (False)
    ones = data[data.columns] == 1
    for sample, series in ones.iterrows():
        # filter out the mutations labeled as False, iterating through the rows
        starts_with_true = series[series == True].index

        # add one to any (mutation, sample) coordinates that had a one in the
        # original data frame
        if sample.startswith('NC'):
            top_ten.loc[starts_with_true, 'NC'] += 1
        elif sample.startswith('C'):
            top_ten.loc[starts_with_true, 'C'] += 1

    # calculate statistics about cancer and non-cancer mutations/samples
    division = lambda row: 1.0 if (row['%C'] / row['%NC'] == np.inf) else row['%C'] / row['%NC'] 
    top_ten['%C'] = top_ten.apply(lambda row: row['C'] / row['T'], axis=1)
    top_ten['%NC'] = top_ten.apply(lambda row: row['NC'] / row['T'], axis=1)
    top_ten['%C-%NC'] = top_ten.apply(lambda row: row['%C'] - row['%NC'], axis=1)
    top_ten['%C/%NC'] = top_ten.apply(division, axis=1)

    get_top_ten(top_ten)
    

################################################################################
# Week 3
################################################################################
def confusion_matrices(data):
    """Generate confusion matrices for the data set"""
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

    
def bar_charts(confusion_matrices):
    """Display bar charts showing the tp, fp, tn, and fn results for two genes """
    # need 2 axes: one for each gene
    # each axis shows two stacked bar charts

    # gather information for charts, e.g. labels
    titles = [confusion_matrices[0].columns.values[0], confusion_matrices[1].columns.values[0]]
    labels = ['Positives', 'Negatives']

    # TODO: hardcoded for two genes: make more dynamic in future
    # first gene: RNF
    rnf = confusion_matrices[0]
    rnf_tp = len(rnf[rnf.values == 'tp'])
    rnf_fp = len(rnf[rnf.values == 'fp'])
    rnf_tn = len(rnf[rnf.values == 'tn'])
    rnf_fn = len(rnf[rnf.values == 'fn'])
    
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    b0 = ax0.bar('Positives', rnf_tp, width=0.35, label='true positives')
    b1 = ax0.bar('Positives', rnf_fp, bottom=rnf_tp, width=0.35, label='false positives')
    b2 = ax0.bar('Negatives', rnf_tn, width=0.35, label='true negatives')
    b3 = ax0.bar('Negatives', rnf_fn, bottom=rnf_tn, width=0.35, label='false negatives')

    ax0.bar_label(b0, label_type='center')
    ax0.bar_label(b1, label_type='center')
    ax0.bar_label(b2, label_type='center')
    ax0.bar_label(b3, label_type='center')

    ax0.set_ylabel('Count')
    ax0.set_title(titles[0])
    ax0.legend()

    # donut chart
    ax2.pie([rnf_tp, rnf_fp, rnf_tn, rnf_fn], labels=['true positive','false positive','true negative','false negative'], autopct='%.1f%%')

    # second gene: tp
    tp53 = confusion_matrices[1]
    tp53_tp = len(tp53[tp53.values == 'tp'])
    tp53_fp = len(tp53[tp53.values == 'fp'])
    tp53_tn = len(tp53[tp53.values == 'tn'])
    tp53_fn = len(tp53[tp53.values == 'fn'])

    b4 = ax1.bar('Positives', tp53_tp, width=0.35, label='true positives')
    b5 = ax1.bar('Positives', tp53_fp, bottom=tp53_tp, width=0.35, label='false positives')
    b6 = ax1.bar('Negatives', tp53_tn, width=0.35, label='true negatives')
    b7 = ax1.bar('Negatives', tp53_fn, bottom=tp53_tn, width=0.35, label='false negatives')

    ax1.bar_label(b4, label_type='center')
    ax1.bar_label(b5, label_type='center')
    ax1.bar_label(b6, label_type='center')
    ax1.bar_label(b7, label_type='center')

    ax1.set_ylabel('Count')
    ax1.set_title(titles[1])
    ax1.legend()

    ax3.pie([tp53_tp, tp53_fp, tp53_tn, tp53_fn], labels=['true positive','false positive','true negative','false negative'], autopct='%.1f%%')


def find_best(matrices):
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
    sorted_diffs = sorted(statistics, key=lambda x: statistics[x][0], reverse=True)[:10]
    sorted_percents = sorted(statistics, key=lambda x: statistics[x][0], reverse=True)[:10]
    print('========== Top 10 mutations by difference (TP - FP) ==========')
    for i in sorted_diffs:
        print(i, '\t', statistics[i])

    print()
    print('========== Top 10 mutaions by percent difference (%TP - %FP) ==========')
    for i in sorted_percents:
        print(i, '\t', statistics[i])
    return (sorted_diffs, sorted_percents)

################################################################################
# Week 4
################################################################################

# pseudo decision tree (I would like to do this with an actual binary tree)
def classify(mutation_of_interest):
    positive = []
    negative = []
    samples = mutation_of_interest.index

    for idx, val in enumerate(mutation_of_interest.values):
        positive.append(samples[idx]) if (val == 1) else negative.append(samples[idx])

    # print()
    # print("========== Positives ==========")
    # print(positive)

    # print()
    # print("========== Negatives ==========")
    # print(negative)

def main():
    global number_of_mutations_per_sample
    global number_of_samples_per_mutation
    global samples
    global mutations

    data = pd.read_csv('./mutations.csv', index_col='Unnamed: 0')
    number_of_mutations_per_sample = data.agg(sum, axis=1)
    number_of_samples_per_mutation = data.agg(sum)
    samples = data.index
    mutations = data.columns

    args = []
    optlist = []
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        optlist, args = getopt.getopt(args, 'pem', ['plot', 'explore', 'matrix'])
    else:
        del args
        del optlist

    if len(sys.argv) < 2:
        explore_data(data, number_of_mutations_per_sample, number_of_samples_per_mutation, samples, mutations)
        plot_data(data, number_of_mutations_per_sample, number_of_samples_per_mutation, samples, mutations)
        confusion_matrices(data)
    else:
        if  optlist and ( ('--explore','') in optlist or ('-e', '') in optlist):
            # only need to explore the data
            explore_data(data, number_of_mutations_per_sample, number_of_samples_per_mutation, samples, mutations)
        if optlist and ( ('--plot', '') in optlist or ('-p', '') in optlist):
            # only need to plot the data
            plot_data(data, number_of_mutations_per_sample, number_of_samples_per_mutation, samples, mutations)
        if optlist and ( ('--matrix', '') in optlist or ('-m', '') in optlist):
            # only generate confusion matrices
            matrix(data)
            # matrices = confusion_matrices(data)
            # (top_diff, top_percent_diff) = find_best(matrices)

            # MUTS_LIST = ['RNF43_GRCh38_17:58357800-58357800_Frame-Shift-Del_DEL_C-C--', 'TP53_GRCh38_17:7675088-7675088_Missense-Mutation_SNP_C-T-T_C-C-T']
            # genes_of_interest = [pd.DataFrame(columns=[MUTS_LIST[0]], index=data.index, data=matrices[MUTS_LIST[0]].values),\
            #                      pd.DataFrame(columns=[MUTS_LIST[1]], index=data.index, data=matrices[MUTS_LIST[1]].values)]
            # bar_charts(genes_of_interest)
            
            

    plt.show()


def matrix(data):
    matrices = confusion_matrices(data)
    (top_diff, top_percent_diff) = find_best(matrices)
    
    MUTS_LIST = ['RNF43_GRCh38_17:58357800-58357800_Frame-Shift-Del_DEL_C-C--', 'TP53_GRCh38_17:7675088-7675088_Missense-Mutation_SNP_C-T-T_C-C-T']
    genes_of_interest = [pd.DataFrame(columns=[MUTS_LIST[0]], index=data.index, data=matrices[MUTS_LIST[0]].values),\
                         pd.DataFrame(columns=[MUTS_LIST[1]], index=data.index, data=matrices[MUTS_LIST[1]].values)]
    bar_charts(genes_of_interest)

    classify_mutation = data[top_diff[0]]
    classify(classify_mutation)

if __name__ == '__main__':
        main()
