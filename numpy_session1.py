# coding: utf-8

import pandas as pd
import numpy as np
data = pd.read_csv('mutations.csv', index_col='Unnamed: 0')
data
pos = data.apply(lambda x: x[x==1].index, axis=0)
pos
pos
pos2 = np.where(data[data==1])
pos2
pos2 = np.where(data[data==1].index)
pos2
frame_columns = ['n(t_l)', 'n(t_r)', 'n(t_l, C)', 'n(t_l, NC)', 'n(t_r, C)', \
                        'n(t_r, NC)', 'P_l', 'P_r', 'P(C | t_l)', 'P(NC | t_l)', \
                        'P(C | t_r)', 'P(NC | t_r)', '2*P_l*P_r', 'Q', 'Phi(s, t)']
selection_frame = pd.DataFrame(columns=frame_columns)
selection_frame
classified_positive
get_ipython().run_line_magic('whos', '')
pos
selection_frame['n(t_r)'] = np.where(pos < 1, 1, len(pos))
pos
selection_frame['n(t_r)'] = np.where(pos.index < 1, 1, len(pos))
pos
selection_frame['n(t_r)'] = np.where(pos.size < 1, 1, len(pos))
selection_frame['n(t_r)']
selection_frame
selection_frame
selection_frame['n(t_l)'] = pos.apply(lambda x: 1 if (x.size < 1) else x.size)
selection_frame
p_l = np.apply_along_axis(lambda x: x['n(t_l)'] / data.shape[0], 1, selection_frame)
p_l = np.apply_along_axis(lambda x: x['n(t_l)'] / data.shape[0], 1, selection_frame['P_l'])
p_l = np.apply_along_axis(lambda x: x['n(t_l)'] / data.shape[0], 0, selection_frame['P_l'])
selection_frame
select = np.array([0 for i in range(15)])
select
select[0]
select = np.array([ [for k in range(15)] for i in range(data.shape[0])]

)
select = np.array([ [for k in range(15)] for i in range(data.shape[0])])
select = np.array([ [0 for k in range(15)] for i in range(data.shape[0])])
select
select[0][1]
select = np.zeroes((15, data.shape[0]))
select = np.zeros((15, data.shape[0]))
select
pos
dnp = data.to_numpy()
dnp
data
dnp = data.to_numpy(na_value = 1)
data
dnp
dnp
data.drop(index)
data
data.to_numpy()
data.reset_index(drop=True)
data.reset_index(drop=True).to_nump()
data.reset_index(drop=True).to_numpy()
pd.DataFrame({'A': [1,2], 'B': [3,4]})
pd.DataFrame({'A': [1,2], 'B': [3,4]}).to_numpy()
data.to_numpy()
data
del  data
data = pd.read_csv('mutations.csv', index_col='Unnamed: 0')
data
data
data[data==1]
data[data==1].to_numpy()
data[data==1].to_numpy(na_value=2)
dnp = data.to_numpy()
dnp
data
data.apply(lambda x: x[x==1].index)
data.apply(lambda x: x[x==1].index, axis=0)
np.argmax()
np.argmax(dnp[0])
selection_frame
np.apply_over_axes(lambda x: x['n(t_l)'] / 230, selection_frame['n(t_l)'])
np.arange(24).reshape(2,3,4)
np.apply_over_axes(lambda x: x['n(t_l)'] / 230, selection_frame['n(t_l)'], 0)
np.apply_over_axes(lambda: x['n(t_l)'] / 230, selection_frame['n(t_l)'], 0)
np.apply_over_axes(lambda x,y: x['n(t_l)'] / 230, selection_frame['n(t_l)'], 0)
np.apply_over_axes(lambda a,axis: a['n(t_l)'] / 230, selection_frame['n(t_l)'], 0)
def psubl(x, axis=0):
    l = lambda x: x['n(t_l)'] / 230
    return l(x)
    
np.apply_over_axes(psubl, selection_frame['n(t_l)'], 0)
temp = np.array([[1,0],[2,0], [3,0]])
temp
temp[0]
temp = np.array([1,2,3], [0,0,0])
temp = np.array([[1,2,3], [0,0,0]])
temp
temp[0]
temp[1] = 2 * temp[0]
temp
temp.transpose()
import random
del random
np.random.randn(100)
np.random.randn(1000)
np.random.randn(100)
pd.DataFrame(np.random.randn(100))
x = np.random.randn(1000)
tempdf = pd.DataFrame(x)
tempdf
tempdf = pd.DataFrame(x, columns=['x', 'y'])
select
select[0]
select[:,0]
select[:,0]=1
select
select[:,0]
pos
a = np.array([1,1,3,4,1],[6,5,4,1,1])
a = np.array([[1,1,3,4,1],[6,5,4,1,1]])
a
np.argwhere(a==1)
pos
np.array(pos)
np.array(pos.size)
np.array(pos)
pos = np.array(pos)
pos
pos[:].size
len(pos[:])
pos[:,0]
pos[:,0].size
pos
pos[np.newaxis, :]
pos[:, np.newaxis]
pos = pos[:, np.newaxis]
pos[:]
pos[:,0]
pos
pos[0]
pos[0, :]
pos[:,0]
pos[:,0].size
pos2 = pos
pos2
pos2[:,0]
pos2[:,0] = pos2[:,0].size
pos2
del pos2
pos
del pos
pos = data.apply(lambda x: x[x==1].index)
pos
pos = np.array(pos)
pos
pos[:, np.newaxis]
pos[:, np.newaxis].apply_along_axis(lambda x: x.size, 0, pos[:])
pos = pos[:, np.newaxis]
pos
np.apply_along_axis(lambda x: x.size, 0, pos)
np.apply_along_axis(lambda x: x.size, (0,0), pos)
np.apply_along_axis(lambda x: x.size, 1, pos)
pos
pos = data.apply(lambda x: x[x==1].index)
pos = np.array(pos)
pos
pos[:, np.newaxis]
pos[np.newaxis,:]
pos
np.apply_along_axis(lambda x: x.size, 0, pos)
pos[pos.size]
pos[:].size
pos[:,0].size
pos[0].size
pos
np.vectorize(lambda x: x.size)
size_func = np.vectorize(lambda x: x.size)
size_func(pos)
pos = size_func(pos)
pos
select
select[0,0]
select[0]
select[0,:]
select[:,0]
select[0,:]
select[:,0]
select[:,0] = pos
select[:,0] = pos.T
pos
pos.shape
select.shape()
select.shape
select
select = np.zeros((15, data.shape[1]))
select.shape
select[:,0] = pos
select[:,0] = pos.T
pos.T.shape
pos
pos.shape
select.shape
select[0,:] = pos
select
pos
data[data==1]
np.array(data[data==1])
data[data==1].index
np.array(data[data==1].index)
np.array(data[data==1].index.size)
data[data==1].index
data.apply(lambda x: x[x==1])
data.apply(lambda x: x[x==1].index)
np.array(data.apply(lambda x: x[x==1].index))
pos2 = np.array(data.apply(lambda x: x[x==1].index))
pos2 = size_func(pos2)
pos2
select
select[0,:]
select[1,:] = select[0,:] / 230
select
get_ipython().run_line_magic('help', 'save')
get_ipython().run_line_magic('info', '')
get_ipython().run_line_magic('help', '')
get_ipython().run_line_magic('magic', '')
get_ipython().run_line_magic('save', 'numpy_session1 0-218')
