
#%%

import pandas as pd
import numpy as np
import random
from typing import Tuple
import sklearn.datasets
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from SVM_class import SVM
from sklearn.metrics import classification_report
from dataset import spirals
from dataset import blobs
from auxillary_functions import get_accuracy

sd = 0.05
means = [[0,0],[1,0],[2,0]]
cov1 = [[sd, 0], [0, sd]]
cov2 = [[sd, 0], [0, sd]]
cov3 = [[sd, 0], [0, sd]]
covs = [cov1, cov2, cov3]

'''

sd = 0.05
means = [[0,0],[1,0]]
cov1 = [[sd, 0], [0, sd]]
cov2 = [[sd, 0], [0, sd]]
covs = [cov1, cov2]

'''

df = blobs(n=50, center = means, cov = covs)

label = df['class']
colors = ['red','blue','cyan']
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=label, cmap=matplotlib.colors.ListedColormap(colors))




#%%

'''

df = spirals(n=500,cycles = 2, sd = 0.25)

'''



#%%

X1 = list(df.iloc[:,0])
X2 = list(df.iloc[:,1])
X = np.array([(X1[i],X2[i]) for i in range(0,len(X1))])
Y = list(df['class'])


X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=.3)#, random_state=1)
#Y_train = np.squeeze(Y_train, axis = 1)

#%%



svm = SVM(kernal='gaussian')




#%%

'''
implement one versus all classification
'''



def one_v_all(
    C: float,
    tol: float,
    max_passes: int,
    n_classes: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    )-> np.ndarray:



    fits = []
    fX = []

    #n = X_test.shape[0]
    #scores = np.zeros((n,n_classes))
    n_classes = len(np.unique(y_train))

    for i in range(n_classes): 
   
        binary = 1.*(y_train == i) - 1.*(y_train != i)
        fits.append(svm.fit(x_train, binary, C, tol, max_passes))

        fX.append(svm.predict_2(x_test))
    return(fX)
# %%

'''

fX = one_v_all(
    C = 1.0,
    tol = 0.01,
    max_passes = 50,
    n_classes = 3,
    x_train = X_train,
    y_train = Y_train)



pred = np.unique(Y_train)[np.argmax(fX, axis = 0)]


label = pred
colors = ['red','blue','cyan']
plt.scatter(X_test[:,0], X_test[:,1], c=label, cmap=matplotlib.colors.ListedColormap(colors))

#print(classification_report(Y_test, pred, target_names = ['red', 'blue','cyan']))


acc = get_accuracy(pred, Y_test)


'''


#%%

'''
select C using n-fold cross validation

'''


def n_fold_cv(
    n_folds: int, # number of folds
    c_low: float, # lower c bound
    c_high: float, # upper c bound
    c_step: float, # step size for c
    X: np.ndarray, # predictor data
    Y: np.ndarray, # response data
) -> pd.DataFrame:


    rows = []
    acc_list = []

    for C in np.arange(c_low, c_high, c_step):


        for n in range(n_folds): 

            # train test split
            X_train, X_test, Y_train, Y_test = train_test_split(
                np.array(X),
                np.array(Y),
                test_size=.3)#, random_state=1)



            # call the on_v_all

            fX = one_v_all(
            C = C,
            tol = 0.01,
            max_passes = 20,
            n_classes = 3,
            x_train = X_train,
            y_train = Y_train,
            x_test = X_test)

        

            pred = np.unique(Y_test)[np.argmax(fX, axis = 0)]


            # get accuracy and append to list
            acc = get_accuracy(pred, Y_test)

            print('accuracy', acc)

            acc_list.append(acc)


        avg_acc = np.average(acc_list)

        rows.append([C, avg_acc])

    df = pd.DataFrame(rows, columns=["C", "accuracy"])

    return(df)


#%%

'''

print('stop here')

df = n_fold_cv(
    n_folds = 5,
    c_low = 1.0,
    c_high = 1.2,
    c_step = 0.10,
    X = X, 
    Y = Y
)

'''



#%%


fX = one_v_all(
    C = 1.50,
    tol = 0.01,
    max_passes = 50,
    n_classes = 3,
    x_train = X_train,
    y_train = Y_train,
    x_test = X_test)


#%%

pred = np.unique(Y_test)[np.argmax(fX, axis = 0)]



#%%

acc = get_accuracy(pred, Y_test)

print('accuracy', acc)

#%%
'''


use linear, gaussian, and other mentioned kernal 


try on gaussian and spiral dataset

try on real life dataset

1.
2.
3.

'''
# %%
