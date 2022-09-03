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
    """Allow the user to click on a data point on the scatter plot and display its coordinates"""
    ind = event.ind
    if event.mouseevent.inaxes:
        ax = event.mouseevent.inaxes
        title = ax.get_title()
        if title == 'Number of mutations per sample':
            print('ax0: ', number_of_mutations_per_sample[ind], samples[ind])
        else:
            print('ax1: ', number_of_samples_per_mutation[ind], mutations[ind])

def plot_data(data, mut_per_sample, samples_per_mut, samples_local, muts):
    """Create two scatter plots, plot the data, and display it.  Scatter plots can be interacted with by the user
       and because of this will have no labels on the x-axes"""
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

    plt.show()

def get_top_ten(top_ten):
    """Sort each column of top_ten and print the ten highest values and their corresponding mutations for each"""
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
        optlist, args = getopt.getopt(args, 'pe', ['plot', 'explore'])
    else:
        del args
        del optlist

    if len(sys.argv) < 2:
        explore_data(data, number_of_mutations_per_sample, number_of_samples_per_mutation, samples, mutations)
        plot_data(data, number_of_mutations_per_sample, number_of_samples_per_mutation, samples, mutations)
    if  optlist and ( ('--explore','') in optlist or ('-e', '') in optlist):
        # only need to explore the data
        explore_data(data, number_of_mutations_per_sample, number_of_samples_per_mutation, samples, mutations)
    if optlist and ( ('--plot', '') in optlist or ('-p', '') in optlist):
        # only need to plot the data
        plot_data(data, number_of_mutations_per_sample, number_of_samples_per_mutation, samples, mutations)

if __name__ == '__main__':
        main()
