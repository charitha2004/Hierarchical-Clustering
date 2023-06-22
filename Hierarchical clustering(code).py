#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Hierarchical Clustering 


# In[2]:


#import the libraries
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


#import dataset
dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:, :].values
X


# In[7]:


#Dendrogram to find the optimal number of clusters 
import scipy.cluster.hierarchy as sch
dendrogram =sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.show()


# In[17]:


#train the model
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters = 7)
y_hc=clustering.fit_predict(X)


# In[18]:


y_hc


# In[19]:


#visualizing the cluster
plt.scatter(X[y_hc == 0 , 0],X[y_hc == 0 , 1], c ='red', label='clusrer 1')
plt.scatter(X[y_hc == 1 , 0],X[y_hc == 1 , 1], c ='green', label='clusrer 2')
plt.scatter(X[y_hc == 2 , 0],X[y_hc == 2 , 1], c ='pink', label='clusrer 3')
plt.scatter(X[y_hc == 3 , 0],X[y_hc == 3 , 1], c ='blue', label='clusrer 4')
plt.scatter(X[y_hc == 4 , 0],X[y_hc == 4 , 1], c ='orange', label='clusrer 5')
plt.scatter(X[y_hc == 5 , 0],X[y_hc == 5 , 1], c ='orange', label='clusrer 6')
plt.scatter(X[y_hc == 6 , 0],X[y_hc == 6 , 1], c ='orange', label='clusrer 7')
plt.title("Cluster of customers ")
plt.xlabel("Annual income(k$)")
plt.ylabel("spending score(1-100)")
plt.legend()
plt.show()


# In[ ]:




