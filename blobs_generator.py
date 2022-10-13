
#%%
import sklearn.datasets
import pandas as pd
#%%

blobs_data = sklearn.datasets.make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=2.0, center_box=(- 10.0, 10.0), shuffle=True, random_state=None, return_centers=False)


df2 = pd.DataFrame.from_dict(blobs_data)


#%%
x1 = list(df2.iloc[0,0][:,0])
x2 = list(df2.iloc[0,0][:,1])
X = [(x1[i],x2[i]) for i in range(0,len(x1))]
Y = list(df2.iloc[1,0])
# %%


import matplotlib.pyplot as plt
import matplotlib

#label = Y
colors = ['red','blue']
plt.scatter(np.array(x1), np.array(x2), c=df2.iloc[1,0], cmap=matplotlib.colors.ListedColormap(colors))
# %%
