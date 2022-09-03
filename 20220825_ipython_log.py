# IPython log file

import pandas as pd
mutations = pd.read_csv('mutations.csv')
mutations
mutations.loc['C1',:]
mutations
mutations.columns
mutations.index
mut = pd.read_csv('mutations.csv', index_col=False)
mut
mut = pd.read_csv('mutations.csv', index_col='Unnamed: 0')
mut
mut.columns
mut.index
get_ipython().run_line_magic('logstart', '')
mut = pd.read_csv('mutations.csv', index_col='Unnamed: 0')
mut.loc['C1', :]
mut.loc['C1', :].agg(sum)
mut.loc['NC1', :].agg(sum)
for i in mut.index:
    print(mut.loc[i, :])
    
avg = 0
for i in mut.index:
    avg += mut.loc[i, :].agg(sum)
    
avg
avg / 230
minMax = []
for i in mut.index:
    minMax.append( mut.loc[i, :].agg(sum))
    
    
minMax
min(minMax)
max(minMax)
mut.loc[:, 'BRAF']
mut.columns
list(mut.columns)
for i in list(mut.columns):
    if 'BRAF' in i:
        print(i)
        
braf = ''
for i in list(mut.columns):
    if 'BRAF' in i:
        braf = i
        
        
braf
mut.loc[:, braf]
mut.loc[:, braf].agg(sum)
kras = ''
for i in list(mut.columns):
    if 'KRAS' in i:
        kras = i
        
        
kras
mut.loc[:, kras].agg(sum)
avgMutations = []
del avgMutations
avgMuts = 0
for i in mut.columns:
    avgMuts += mut.loc[:, i].agg(sum)
    
avgMuts
avgMuts / 3816
minMaxMuts = []
for i in mut.columns:
    minMaxMuts.append( mut.loc[:, i].agg(sum))
    
min(minMaxMuts)
max(minMaxMuts)
get_ipython().run_line_magic('logstop', '')
