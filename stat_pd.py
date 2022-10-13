

'''
Handles dataframe filtering and sorting
'''

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Tuple

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer




def split_num_cat(
    file_name: str,
    )-> Tuple[np.ndarray,np.ndarray]:

    '''
    splits numerical and categorical attributes
    '''

    df = pd.read_csv(file_name)
    df = df.drop(df.columns[0], axis = 1)
    df_num = df.loc[:,df.dtypes!=np.object]
    df_cat = df.loc[:,df.dtypes==np.object]

    return(df_num, df_cat)


def filter_corr(
    df_num: pd.DataFrame,
    )->pd.DataFrame:

    '''
    removes correlated features
    '''
    print('df_num', type(df_num))
    print('df_num.mean', type(df_num.mean()))

    norm_df_num = (df_num - df_num.mean())/df_num.std()
    correlation_matrix = norm_df_num.corr()
    correlated_features = []
    for i in np.arange(0,len(correlation_matrix.columns)):
        for j in np.arange(0,len(correlation_matrix.columns)):
            #print(correlation_matrix.iloc[i,j])
            #print(i,j)
            if abs(correlation_matrix.iloc[i,j])>0.9:
                if i!=j:
                    print(i,j)
                    colname = correlation_matrix.columns[min(i,j)]
                    print(colname)
                    correlated_features.append(colname)
    corr_feat = list(set(correlated_features))
    df_num = df_num.drop(corr_feat, axis = 1)
    return(df_num)



def cat_to_num(
    df_cat: np.ndarray
    )-> np.ndarray:

    encoder = OrdinalEncoder()

    result = encoder.fit_transform(df_cat)

    result_df = pd.DataFrame(result)
    return(result_df)
