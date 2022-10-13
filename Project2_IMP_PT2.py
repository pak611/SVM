
#%%

from tkinter import Label
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
import os 
from stat_pd import split_num_cat
from stat_pd import cat_to_num

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import seaborn as sns

from sklearn.feature_extraction import text

#%%

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

'''


#%%

os.chdir('/Users/patrickkampmeyer/Dropbox/Ph.D/Classes/Spring_2022/Machine_Learning/Project_2/dataset/')


#----------------------------------- ADULT DATASET -------------------------------------------------

df = pd.read_csv('adult.data')

#%%

df.columns = ['age', 'workclass', 'fnlwgt', 'education_cat', 'education_cont', 
                'marital_status', 'occupation', 'relationship', 'race', 'sex',
                'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'wage']




#%%

obj_df = df.select_dtypes(include=['object']).copy()

#%%

df_cat = cat_to_num(obj_df)
df_cat.columns = obj_df.columns



#%%

df_num = df.drop(df[obj_df.columns], axis = 1)


df = pd.concat([df_num, df_cat], axis = 1)


#%%


#sns.pairplot(df)


#%%


scaler = StandardScaler()

X = df.drop('wage', axis = 1)
Y = df['wage']

X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

#%%

pca = PCA(n_components = 2)

reduced_X = pd.DataFrame(pca.fit_transform(X), columns = ['PC1', 'PC2'])

#%%
reduced_X['cluster'] = Y

plt.figure(figsize=(15,10))
plt.scatter(reduced_X[reduced_X['cluster']==0].loc[:,'PC1'], reduced_X[reduced_X['cluster']==0].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==1].loc[:,'PC1'], reduced_X[reduced_X['cluster']==1].loc[:,'PC2'])
plt.title("Wage")
plt.xlabel("PC1")
plt.ylabel("PC2")

#%%



X_train, X_test, Y_train, Y_test = train_test_split(np.array(reduced_X), np.array(Y), test_size=.3)#, random_state=1)
# %%


svm = SVM(kernal='linear')

#%%

fX = svm.one_v_all(
    C = 1.00,
    tol = 0.01,
    max_passes = 10,
    n_classes = 2,
    x_train = X_train,
    y_train = Y_train,
    x_test = X_test)


#%%

pred = np.unique(Y_test)[np.argmax(fX, axis = 0)]



#%%

acc = get_accuracy(pred, Y_test)

print('accuracy', acc)

# %%


#----------------------------------- AUTO-MPG DATASET -------------------------------------------------

df = pd.read_csv('auto-mpg.csv')

#%%

df.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration',
                'model_year','origin','car_name']




#%%

obj_df = df.select_dtypes(include=['object']).copy()

#%%

df_cat = cat_to_num(obj_df)
df_cat.columns = obj_df.columns



#%%

df_num = df.drop(df[obj_df.columns], axis = 1)


df = pd.concat([df_num, df_cat], axis = 1)


#%%


#sns.pairplot(df)


#%%


scaler = StandardScaler()

X = df.drop('origin', axis = 1)
Y = df['origin']

X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

#%%

pca = PCA(n_components = 2)

reduced_X = pd.DataFrame(pca.fit_transform(X), columns = ['PC1', 'PC2'])

#%%
reduced_X['cluster'] = Y

plt.figure(figsize=(15,10))
plt.scatter(reduced_X[reduced_X['cluster']==0].loc[:,'PC1'], reduced_X[reduced_X['cluster']==0].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==1].loc[:,'PC1'], reduced_X[reduced_X['cluster']==1].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==2].loc[:,'PC1'], reduced_X[reduced_X['cluster']==2].loc[:,'PC2'])
plt.title("Origin")
plt.xlabel("PC1")
plt.ylabel("PC2")

#%%



X_train, X_test, Y_train, Y_test = train_test_split(np.array(reduced_X), np.array(Y), test_size=.3)#, random_state=1)
# %%


svm = SVM(kernal='linear')

#%%

fX = svm.one_v_all(
    C = 1.25,
    tol = 0.01,
    max_passes = 20,
    n_classes = 3,
    x_train = X_train,
    y_train = Y_train,
    x_test = X_test)


#%%

pred = np.unique(Y_test)[np.argmax(fX, axis = 0)]



#%%

acc = get_accuracy(pred, Y_test)

print('accuracy', acc)

# %%


#----------------------------------- CANCER DATASET -------------------------------------------------


#%%
df = pd.read_csv('breast-cancer-wisconsin.data')

#%%

df.columns = ['ID','CT','U_SIZE','U_SHAPE','MA','SECS','BN',
            'BC','NN', 'MITOSIS', 'CLASS']




#%%

obj_df = df.select_dtypes(include=['object']).copy()

#%%

df_cat = cat_to_num(obj_df)
df_cat.columns = obj_df.columns



#%%

df_num = df.drop(df[obj_df.columns], axis = 1)


df = pd.concat([df_num, df_cat], axis = 1)


#%%


#sns.pairplot(df)


#%%


scaler = StandardScaler()

X = df.drop('CLASS', axis = 1)
Y = df['CLASS']

X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

#%%

pca = PCA(n_components = 2)

reduced_X = pd.DataFrame(pca.fit_transform(X), columns = ['PC1', 'PC2'])

#%%
reduced_X['cluster'] = Y

plt.figure(figsize=(15,10))
plt.scatter(reduced_X[reduced_X['cluster']==2].loc[:,'PC1'], reduced_X[reduced_X['cluster']==2].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==4].loc[:,'PC1'], reduced_X[reduced_X['cluster']==4].loc[:,'PC2'])
plt.title("Class")
plt.xlabel("PC1")
plt.ylabel("PC2")

#%%



X_train, X_test, Y_train, Y_test = train_test_split(np.array(reduced_X), np.array(Y), test_size=.3)#, random_state=1)
# %%


svm = SVM(kernal='linear')

#%%

fX = one_v_all(
    C = 1.0,
    tol = 0.01,
    max_passes = 20,
    n_classes = 2,
    x_train = X_train,
    y_train = Y_train,
    x_test = X_test)


#%%

pred = np.unique(Y_test)[np.argmax(fX, axis = 0)]



#%%

acc = get_accuracy(pred, Y_test)

print('accuracy', acc)




#----------------------------------- MNIST DATASET -------------------------------------------------


#%%
os.chdir('/Users/patrickkampmeyer/Dropbox/Ph.D/Classes/Spring_2022/Machine_Learning/Project_2/dataset/')



### Preprocessing
def preprocess(data_set):
    # load
    data_set = np.array(pd.read_csv('mnist_train.csv'))

    Y = data_set[:,0]

    X = data_set[:,1:]
    
    # we need test data to be in the data to get a score... so lets divide the train.csv dataset up into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

   # y_train_origin = train_split[:,1]
   # y_train_start = train_split[:,1]
   # X_test = test_split[:,1]

    # preprocessing

    # transform to numeric label
    
    le = preprocessing.LabelEncoder()
    le.fit([0,1,2,3,4,5,6,7,8,9])
    y_train = le.transform(y_train)
    # add in here
    y_test = le.transform(y_test)
    class_num = len(np.unique(y_train))

    # Transform from strings into counts of strings and then normalize
    
    vectorizer = text.TfidfVectorizer(max_features = 785, binary = True, stop_words = text.ENGLISH_STOP_WORDS)
    normalizer_train = preprocessing.Normalizer()
    
    # UNCOMMENT WHEN USING PIXEL DATA
 #  X_vectors_train = vectorizer.fit_transform(X_train)
#   X_vectors_test = vectorizer.transform(X_test)

    X_vectors_train = normalizer_train.transform(X_train)
    X_vectors_test = normalizer_train.transform(X_test)

    return X_vectors_train, y_train, X_vectors_test, y_test, class_num, le


#%%
x_train, y_train, x_test, y_test, class_num, le = preprocess('mnist_train.csv')

#%%

X = pd.DataFrame(x_train)
Y = pd.DataFrame(y_train)

#%%


scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

#%%

pca = PCA(n_components = 2)

reduced_X = pd.DataFrame(pca.fit_transform(X), columns = ['PC1', 'PC2'])

#%%
reduced_X['cluster'] = Y

plt.figure(figsize=(15,10))
plt.scatter(reduced_X[reduced_X['cluster']==1].loc[:,'PC1'], reduced_X[reduced_X['cluster']==1].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==2].loc[:,'PC1'], reduced_X[reduced_X['cluster']==2].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==3].loc[:,'PC1'], reduced_X[reduced_X['cluster']==3].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==4].loc[:,'PC1'], reduced_X[reduced_X['cluster']==4].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==5].loc[:,'PC1'], reduced_X[reduced_X['cluster']==5].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==6].loc[:,'PC1'], reduced_X[reduced_X['cluster']==6].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==7].loc[:,'PC1'], reduced_X[reduced_X['cluster']==7].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==8].loc[:,'PC1'], reduced_X[reduced_X['cluster']==8].loc[:,'PC2'])
plt.scatter(reduced_X[reduced_X['cluster']==9].loc[:,'PC1'], reduced_X[reduced_X['cluster']==9].loc[:,'PC2'])
plt.title("Number")
plt.xlabel("PC1")
plt.ylabel("PC2")

#%%



X_train, X_test, Y_train, Y_test = train_test_split(np.array(reduced_X), np.array(Y), test_size=.3)#, random_state=1)
# %%


svm = SVM(kernal='linear')

#%%

fX = one_v_all(
    C = 1.0,
    tol = 0.01,
    max_passes = 20,
    n_classes = 9,
    x_train = X_train,
    y_train = Y_train,
    x_test = X_test)


#%%

pred = np.unique(Y_test)[np.argmax(fX, axis = 0)]



#%%

acc = get_accuracy(pred, Y_test)

print('accuracy', acc)