# coding: utf-8
mut.index
for i in mut.index:
    if 'NC' in i:
        nc+=1
        
nc
230 - 210
230 - 120
kras_genes
mut.loc[:, kras_genes]
list(mut.loc[:, kras_genes])
mut.loc[:, kras_genes]
mut.loc[:, kras_genes].iterrows()
for i, j in mut.loc[:, kras_genes].iterrows():
    print(i, ' ', j)
    
mut.columns
for i in mut.columns:
    if 'KRAS' in i:
        print(i)
        
kras_genes
mut[mut[mut.columns.includes('KRAS')]]
mut[mut[list(mut.columns).includes('KRAS')]]
mut[mut[mut.columns.includes('KRAS')]]
kras_genes
for k in kras_genes:
    for j in kras_genes:
        k.isin(j)
        
for k in kras_genes:
    mut.k.isin(mut.k)
    
for k in kras_genes:
    mut[k].isin(mut[k])
    
    
for k in kras_genes:
    print(mut[k].isin(mut[k]))
    
    
    
mut.columns[mut.columns.str.contains(pat='KRAS')]
for k in kras_genes:
    print(mut[k].isin(mut[k]))
    
mut.loc[:, kras_genes].agg(sum)
sum(mut.loc[:, kras_genes].agg(sum))
import numpy as np
np.array(mut.loc[:, kras_genes])
list(np.array(mut.loc[:, kras_genes]))
np.array(mut.loc[:, kras_genes])
mut.loc[:, kras_genes].agg(sum)
for i in mut.loc[:, kras_genes].agg(sum):
    if i > 0:
        print(1)
        
np.array(mut.loc[:, kras_genes])
kras_list = np.array(mut.loc[:, kras_genes])
for i in kras_list:
    sum(i)
    
for i in kras_list:
    print(sum(i))
    
    
for i in kras_list:
    print(sum(i))
x = 0
for i in kras_list:
    x += sum(i)
    
x
x = 0
for i in kras_list:
    f = sum(i)
    if f > 0:
        f = 1
    x += f
    
    
x
get_ipython().run_line_magic('save', 'temp_file.py 80-123')
