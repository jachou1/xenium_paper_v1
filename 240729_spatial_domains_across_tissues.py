import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Let's read in all the csv files with the vectors of fraction of different cell-types per spatial domain
working_dir = '/Users/jacquelinechou/Downloads/'
s1_r1 = pd.read_csv(working_dir + 's1_r1' + '/slide1_region1_kmeans_5_df.csv', index_col= 0)
s1_r3 = pd.read_csv(working_dir + 's1_r3' + '/slide1_region3_kmeans_5_df.csv', index_col= 0)
s1_r2 = pd.read_csv(working_dir + 's1_r2' + '/slide1_region2_kmeans_5_df.csv', index_col= 0)
# Re-run, I set the random_state variable, but I need to confirm it's the same labeling as in .h5ad
s1_r4 = pd.read_csv(working_dir + 's1_r4' + )
s1_r5 = pd.read_csv(working_dir + 's1_r5' + '/slide1_region5_kmeans_5_df.csv', index_col = 0)
s2_r1 = pd.read_csv(working_dir + 's2_r1' + '/slide2_region1_kmeans_5_df.csv', index_col= 0)
s2_r2 = pd.read_csv(working_dir + 's2_r2' + '/slide2_region2_kmeans_5_df.csv', index_col= 0)
s2_r3 = pd.read_csv(working_dir + 's2_r3' + '/slide2_region3_kmeans_5_df.csv', index_col= 0) # np array
# Re-run, need to match up spatial domains
s2_r4 = pd.read_csv(working_dir + '')
s2_r5 = pd.read_csv(working_dir + 's2_r5' + '/slide2_region5_kmeans_5_df.csv', index_col= 0)
s2_r6 = pd.read_csv(working_dir + 's2_r6' + '/slide2_region6_kmeans_5_df.csv', index_col= 0)


# Concatenated df: 55 rows x 8 columns

# Run hierarchical clustering; I wonder how they will cluster

