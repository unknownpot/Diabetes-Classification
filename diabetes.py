# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


# %%
dfx = pd.read_csv('Diabetes_XTrain.csv')
dfy = pd.read_csv('Diabetes_YTrain.csv')
x = dfx.values
y = dfy.values


dfx


# %%
y.reshape((-1, ))


# %%
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))


# %%
def knn(X, Y, Point, k):
    vals = []
    for i in range(X.shape[0]):
        d = dist(X[i], Point)
        vals.append((d, Y[i]))

    vals = sorted(vals)
    vals = vals[:k]

    vals= np.array(vals)
    new_vals = np.unique(vals[:,1],return_counts = True)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred


# %%
dft = pd.read_csv('Diabetes_Xtest.csv')
test = dft.values


# %%
diabetic_list = []
for i in range(test.shape[0]):
    pred = knn(x, y, test[i], 15)
    diabetic_list.append(pred)

array = np.array(diabetic_list)
array = pd.DataFrame(array)
array.to_csv('Solution.csv',index = False)   


# %%



