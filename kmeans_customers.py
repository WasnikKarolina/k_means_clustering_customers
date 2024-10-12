# Suppressing warnings
def warn(*args, **kwargs): 
    pass

import warnings 
warnings.warn = warn 

# Importing libraries 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
from sklearn.preprocessing import StandardScaler 

# Generating synthetic data for clustering 
np.random.seed(0) 
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

plt.scatter(X[:, 0], X[:, 1], marker='.') 
plt.title("GENERATED RAW DATA")
plt.show()

# Performing KMeans clustering (4 clusters)
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12) 
k_means.fit(X)

k_means_labels = k_means.labels_ 
k_means_cluster_centers = k_means.cluster_centers_ 

# Visualizing clustered data with cluster centers
fig = plt.figure(figsize=(6, 4)) 
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

ax = fig.add_subplot(1, 1, 1) 
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
    
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k] 


    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.') 


    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
plt.title("DATA POINTS WITH 4 CLUSTER ASSIGMENTS")
plt.show()

# Fitting KMeans to the customer data (3 clusters)
k_means3 = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means3.fit(X)

fig = plt.figure(figsize=(6, 4)) 
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_)))) 

ax = fig.add_subplot(1, 1, 1) 

for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
plt.title("DATA POINTS WITH 3 CLUSTER ASSIGMENTS")
plt.show()

# Loading and preprocessing customer segmentation data
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

df = cust_df.drop('Address', axis=1) 

X = df.values[:, 1:] 

X = np.nan_to_num(X) 
Clus_dataSet = StandardScaler().fit_transform(X) 

# Fiting KMeans to the customer data 
clusterNum = 3 
k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
k_means.fit(X)
labels = k_means.labels_

df["Clus_km"] = labels 
print("\n")
print("Customer dataset including column Clus_km with customer labels:")
print("\n")
print(df) 
print("\n")

mean_values = df.drop(['Customer Id'], axis=1).groupby('Clus_km').mean() 
print("The mean values of various features for each cluster (label): ")
print("\n")
print(mean_values)

area = np.pi * (X[:, 1])**2 

# Visualizing Age vs Income
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5) 
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.title("AGE VS INCOME")
plt.show()

# Creating a 3D plot for Education, Age, and Income
fig = plt.figure(1, figsize=(8, 6))

ax = fig.add_subplot(111, projection='3d', elev=48, azim=134)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=labels.astype(float), s=50, alpha=0.6) 
plt.title("EDUCATION VS AGE VS INCOME")
plt.show() 