# IPython log file

get_ipython().run_line_magic('logstart', '08312022_ipython_log.py')
import pandas as pd
data = pd.read_csv('./mutations.csv', index_col='Unnamed: 0')
data
top_ten = pd.DataFrame(index=data.columns, data=list(range(1,3816)))
top_ten = pd.DataFrame(index=data.columns, data=list(range(1,3817)))
top_ten
top_ten.shape
top_ten.index
top_ten.columns
top_ten = pd.DataFrame(index=data.columns, data={'column 1',list(range(1,3817))})
top_ten = pd.DataFrame(index=data.columns, data=list(range(1,3817)), columns='column 1')
top_ten = pd.DataFrame(index=data.columns, data=list(range(1,3817)), columns=['column 1'])
top_ten
top_ten = pd.DataFrame(index=data.columns, data=list(range(1,3817)), columns=['column 1', 'col 2'])
top_ten = pd.DataFrame(index=data.columns,  columns=['column 1', 'col 2'])
top_ten
top_ten.loc[:, 'column 1'] = 1
top_ten
temp_sums = data.agg(sum, axis=1)
temp_sums
temp_sums = data.agg(sum, axis=0)
temp_sums
top_ten
top_ten.iloc[0, 0]
map(lambda x: top_ten.iloc[x, 0] = temp_sums.iloc[x, 0], [x for x in range(0, 3816)])
def assign(x):
    top_ten.iloc[x, 0] = temp_sum.iloc[x, 0]
    
map(assign(x), [x for x in range(0, 3816)])
map(assign(), [x for x in range(0, 3816)])
x = [i for i in range(0, 3816)]
map(assign, x)
list(map(assign, x))
temp_sum
temp_sums
def assign(x):
    top_ten.iloc[x, 0] = temp_sums.iloc[x, 0]
    
    
list(map(assign, x))
temp_sums.iloc[0,0]
temp_sums.iloc[0]
def assign(x):
    top_ten.iloc[x, 0] = temp_sums.iloc[x]
        
list(map(assign, x))
y = list(map(assign, x))
y
y.head()
y.shape
top_ten
x
del x
x
map(assign, [x for x in range(0,3816)])
top_ten
def func(x,y,z,a):
    print(x+y)
    print(z)
    print(a)
    
temp = 'Hello'
temp2 = 'World'
map(func, [x for x in range(0,5)], [y for y in range(6,10)], temp, temp2)
list(map(func, [x for x in range(0,5)], [y for y in range(6,10)], temp, temp2))
list(map(func, [x for x in range(0,5)], [y for y in range(6,10)], data, top_ten))
temp_sums
list(map(func, [x for x in range(0,5)], [y for y in range(6,10)], data, temp_sums))
def func(x,y,z,a):
    print(z)
    print(a)
    
list(map(func, [x for x in range(0,5)], [y for y in range(6,10)], data, temp_sums))
get_ipython().run_line_magic('logoff', '')
