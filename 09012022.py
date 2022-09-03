# coding: utf-8
import pandas as pd
data = pd.read_csv('./mutations.csv', index_col='Unnamed: 0')
data
def func(col):
    return data.loc[:, col] >= 1
    
filter(func, data.columns)
list(filter(func, data.columns))
def func(col):
    return True if data.loc[:, col] >= 1 else False
    
    
list(filter(func, data.columns))
data.iloc[:, 0]
def func(x):
    return True if x == 1 else False
    
list(filter(func, data.iloc[:, 0]))
list(filter(func, data.iloc[:, 1]))
data
data.columns
data['PLD4_GRCh38_14:104932802-104932802_Silent_SNP_G-G-A'].str
data['PLD4_GRCh38_14:104932802-104932802_Silent_SNP_G-G-A']
data.pivot_table()
pd.pivot_table(data)
whois
get_ipython().run_line_magic('whos', '')
del func
top_ten = pd.DataFrame(columns=['T', 'C', 'NC'])
top_ten
top_ten = pd.DataFrame(columns=['T', 'C', 'NC'], index=data.columns)
top_ten
top_ten.apply(lambda row: data[row].agg(sum))
top_ten.apply(lambda row: data[row])
top_ten.apply(lambda row: print(data[row]))
data
for i in data.columns:
    print(data[i])
    
for i in data:
    print(i)
    
for i in data:
    print(i.values)
    
    
for i in data:
    data[i].values
    
    
    
    
for i in data:
    data[i].values
    
for i in data:
    data[i].array
    
    
for i in data:
    print(data[i].array)
    
    
    
data.iterrows()
print(data.iterrows())
for i in data.iteritems():
    print(i)
    
for i in data.iteritems():
    print(i)
    
c_sum = 0
nc_sum = 0
samples_per_muts = []
samples_per_muts = data.agg(sum)
samples_per_muts
for i in samples_per_muts:
    print(i)
    
samples_per_muts
samples_per_muts.T
top_ten
top_ten['T'] = samples_per_muts
top_ten
for i in top_ten.iterrows():
    print(i)
    
for i in top_ten['T'].iterrows():
    print(i)
    
    
for i in top_ten['T'].enumerate():
    print(i)
    
for i in top_ten['T'].enum():
    print(i)
    
for i in top_ten['T'].iteritems():
    print(i)
    
for i in data.iteritems():
    print(i)
    
for i in samples_per_muts.iteritems():
    print(i)
    
for i in data.iteritems():
    print(i)
    
for i in data.iteritems():
    for j in i:
        print(j)
            
for i in data.iloc[:,0].iteritems():
    print(i)
data.index
data.loc[data.index == 1 and data.index.startswith('C'), :]
data.loc[[data[data.columns[0]] == 1]]
data.loc[[data[data.columns[0]] == True]]
data[data.columns[0]] == True
data[data.columns[0]] == 1
data[data.columns[0]] 
data[data.columns[1]] 
data.head(n=10)
data[data.columns[1]] 
data[data.columns[2]] 
data[data.columns[3]] 
data[data.columns[4]] 
data[data.columns[5]] 
data[data.columns[24]] 
data[data.columns[24]] == 1
data[[data[data.columns[24]] == 1]]
data[data.columns[24]] == 1
NC = data[data.columns[24]] == 1
NC
NC.to_list()
NC
NC & NC.index.startswith('NC')
NC.index
for i in NC:
    print(i and i.startswith('NC'))
    
get_ipython().run_line_magic('save', '09012022 1-87')
for i in NC:
    i
    
    
for i in NC:
    print(i)
        
for i,k in NC.iteritems:
    print(i, k)
    
        
for i,k in NC.iteritems():
    print(i, k)
    
        
for i,k in NC.iteritems():
    print(i, k)
    
        
for i,k in NC.iteritems():
    print(i and k.startswith('NC'))
    
    
for i,k in NC.iteritems():
    print(k and i.startswith('NC'))
    
for i,k in NC.iteritems():
    print(sum(k and i.startswith('NC')))
    
get_ipython().run_line_magic('save', '-a 09012022 88-96')
top_ten.loc[:, 'C'] = 0
top_ten.loc[:, 'NC'] = 0
for i,k in ones.iterrows():
    starts_with_true = k[k == True].index
    if i.startswith('C'):
        top_ten.loc[starts_with_true, 'C'] += 1
    elif i.startswith('NC'):
        top_ten.loc[starts_with_true, 'NC'] += 1
        
top_ten
