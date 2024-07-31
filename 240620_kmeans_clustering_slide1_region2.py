import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.neighbors import KDTree

concat_ad = sc.read_h5ad('/Users/jacquelinechou/Downloads/240607_cohort_tcell_recluster_reindexed.h5ad')

# Let's just look at slide1_region2 (T/B aggregate heavy, good to test out spatial niche test)
slide1_region2_tissue = concat_ad[concat_ad.obs['tissue'] == 'slide1_r2']

# Create an adjacency matrix for all 141K cells with a radius = 100 um
def generate_neighbor_graph(single_cell_matrix, distance_threshold=100):
    ## Generates the GRAPH for all cells in data within 100 microns of each other
    kdt = KDTree(single_cell_matrix.obsm['spatial'], metric='euclidean')

    # get the nearest 100 meighbors using kdTree
    distances_matrix, neighbor_indices = kdt.query(single_cell_matrix.obsm['spatial'], k=151, return_distance=True)
    distances_matrix, neighbor_indices = distances_matrix[:, 1:], neighbor_indices[:, 1:]  # this removes the first column which is itself
    distances_mask = distances_matrix < distance_threshold
    return distances_matrix, neighbor_indices, distances_mask

distances_matrix, neighbor_graph, distances_mask = generate_neighbor_graph(slide1_region2_tissue)

masked_indices = np.where(distances_mask, neighbor_graph, -1)

# Initialize a list to hold the filtered neighbor indices for each point
filtered_neighbors = []

# Iterate over each spatial point
for indices in masked_indices:
    # Filter out the placeholder entries (-1)
    filtered_indices = indices[indices != -1]
    # Append the filtered indices to the list
    filtered_neighbors.append(filtered_indices)

# Convert the list of arrays to an array of arrays (object dtype)
filtered_neighbors = np.array(filtered_neighbors, dtype=object)

import networkx as nx
import matplotlib.pyplot as plt

# Create a NetworkX graph
G = nx.Graph()

# Add nodes with positions (dummy positions for this example)
positions = {i: tuple(coord) for i, coord in enumerate(slide1_region2_tissue.obsm['spatial'])}
for node, pos in positions.items():
    G.add_node(node, pos=pos)

# Add edges based on distances_mask and neighbor_indices
edge_indices = np.where(distances_mask == 1)
edges = [(i, neighbor_graph[i, k]) for i, k in zip(*edge_indices)]
G.add_edges_from(edges)

# Plot the neighbor connectivity: can't really discern the T/B aggregates, too many other cells nearby
plt.figure(figsize=(30*3, 30*3), dpi = 200)
nx.draw(G, positions, with_labels=False, node_color='lightblue', node_size=1, font_size=10, font_weight='bold', edge_color='gray')
plt.title("Spatial Network: Slide1 Region2")
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()
plt.savefig('/Users/jacquelinechou/Downloads/s1_r2/240619_s1_r2_spatial_network_90x90_200dpi.png');

# Get the labels of the neighbors and calculate the vector of cell-types per cell
# Run k-means clustering on those vectors of cell-types to get structures

# Let's get the cell-type labels per cell using neighbor graph
subset_list = [slide1_region2_tissue.obs['celltype_tier1_merged'].iloc[neighbors].values for neighbors in filtered_neighbors]

celltype_labels = slide1_region2_tissue.obs['celltype_tier1_merged'].unique()

total_count_celltype = np.ones(shape = [len(subset_list), len(celltype_labels)]) * 5000

# Now for each cell, let's calculate the number of occurrences of each cell-type present
# There are cell-types: 'T-cells', 'Sublining fibroblasts', 'Endothelial cells',
#        'Mast cells/ adipocytes', 'Myeloid cells', 'Plasma cells',
#        'Non-plasma B-cells', 'Lining fibroblasts', nan

for i in np.arange(len(subset_list)): # for cell, i
  # print(i)
  for j in np.arange(len(celltype_labels)):  # for each cell-type (9 total includng 'nan')
    # print(j)
    total_count_celltype[i, j] = np.sum(subset_list[i] == celltype_labels[j])

# Let's get the vector of proportions for each cell-type w/i 100 microns  for each cell
fractional_neighbors_vector = total_count_celltype / np.sum(total_count_celltype, axis = 1).reshape(-1, 1)
fractional_neighbors_vector_no_nans = np.nan_to_num(fractional_neighbors_vector)
# Let's run clustering to cluster these vectors of fractional neighbors and see what clusters are returned
# K-means clustering:
# k = 2

from sklearn.cluster import KMeans
# Instantiate the k-means estimator
# kmeans = KMeans(n_clusters=2, random_state = 16)
# # Fit the model to the data
# kmeans.fit(fractional_neighbors_vector_no_nans)
# # Get the cluster centroids
# centroids = kmeans.cluster_centers_
# # Get the cluster labels for each data point
# labels = kmeans.labels_

# Add these spatial neighborhood_k=2 to the anndata object
# slide1_region2_tissue.obs['spatial_neighborhood_k_2'] = labels

# Let's plot the spatial groups identified via k-means clustering
# plt.figure(figsize=(20, 30), dpi= 300)
# for color, group in slide1_region2_tissue.obs.groupby(['spatial_neighborhood_k_2']):
#     idx = color[0]
#     # print(idx)
#     # let's subset the adata to these neighborhoods
#     plt.scatter(x = group['spatial_x'], y = group['spatial_y'], s= 1, c = list(mcolors.TABLEAU_COLORS)[idx], alpha = 0.5,
#                 label= idx)
# plt.legend(loc='upper right', markerscale = 10, fontsize = 20,
#            # bbox_to_anchor = (1.25, 0.5)
# )
# plt.gca().invert_yaxis()
# plt.gca().set_aspect('equal')
# # legend.legendHandles[0]._legmarker.set_markersize(10)
# # legend.legendHandles[0]._legmarker.set_alpha(1)
# plt.savefig('/Users/jacquelinechou/Downloads/s2_r3_test_spatial_plot_k_2.png')

# Let's include the legend to the spatial plot of spatial neighborhoods

# Set the seed when you run KMeans
kmeans5 = KMeans(n_clusters=5, random_state= 16)
# Fit the model to the data
kmeans5.fit(fractional_neighbors_vector_no_nans)
# Get the cluster centroids
centroids5 = kmeans5.cluster_centers_
# Get the cluster labels for each data point
labels5 = kmeans5.labels_

# Let's also save centroids5 so we have this to refer back to
np.save('/Users/jacquelinechou/Downloads/s1_r2/s1_r2_centroids5_kmeans5.npy', centroids5)

# Add these spatial neighborhood_k=2 to the anndata object
slide1_region2_tissue.obs['spatial_neighborhood_k_5'] = labels5

from matplotlib import colors as mcolors
slide1_region2_tissue.obs['spatial_x'] = slide1_region2_tissue.obsm['spatial'][:, 0]
slide1_region2_tissue.obs['spatial_y'] = slide1_region2_tissue.obsm['spatial'][:, 1]

plt.figure(figsize=(20, 30), dpi= 300)
for color, group in slide1_region2_tissue.obs.groupby(['spatial_neighborhood_k_5']):
    idx = color[0]
    # print(idx)
    # let's subset the adata to these neighborhoods
    plt.scatter(x = group['spatial_x'], y = group['spatial_y'], s= 1, c = list(mcolors.TABLEAU_COLORS)[idx], alpha = 0.5,
                label= idx)
plt.legend(loc='upper right', markerscale = 10, fontsize = 20,
           # bbox_to_anchor = (1.25, 0.5)
)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
# legend.legendHandles[0]._legmarker.set_markersize(10)
# legend.legendHandles[0]._legmarker.set_alpha(1)
plt.savefig('/Users/jacquelinechou/Downloads/s1_r2/s1_r2_test_spatial_plot_k_5.png')

# Let's make the stacked barplot of the spatial clusters identified using kmeans
spatial_neighborhoods_df = pd.DataFrame(centroids5, columns = celltype_labels)

spatial_neighborhoods_df['Spatial_neighborhood'] = ['Spatial cluster 0', 'Spatial cluster 1', 'Spatial cluster 2',
                                                    'Spatial cluster 3', 'Spatial cluster 4']
# Save the dataframe as a csv file with the cell-type proportion per spatial group
spatial_neighborhoods_df.to_csv('/Users/jacquelinechou/Downloads/s1_r2/slide1_region2_kmeans_5_df.csv')

# Stacked barplot
spatial_neighborhoods_df.plot(x = 'Spatial_neighborhood', kind = 'bar', stacked=True)
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 0.5))
plt.savefig('/Users/jacquelinechou/Downloads/s1_r2/240619_s1_r2_celltype_composition_barplot_kmeans_5.png', bbox_inches='tight')

# This tissue has significant lining hyperplasia (3), but no T/B-cell aggregates
# Curious to know what changing the values of k would do in terms of being able to pick up regions of different layers
k = 2

# Instantiate the k-means estimator
kmeans2 = KMeans(n_clusters=2, random_state = 16)
# Fit the model to the data
kmeans2.fit(fractional_neighbors_vector_no_nans)
# Get the cluster centroids
centroids2 = kmeans2.cluster_centers_
# Get the cluster labels for each data point
labels2 = kmeans2.labels_

# Let's also save centroids5 so we have this to refer back to
np.save('/Users/jacquelinechou/Downloads/s2_r1/s2_r1_centroids2_kmeans2.npy', centroids2)

spatial_nb_2 = pd.DataFrame(centroids2, columns = celltype_labels)

spatial_nb_2['Spatial_neighborhood'] = ['Spatial cluster 0', 'Spatial cluster 1']
# Save the dataframe as a csv file with the cell-type proportion per spatial group
spatial_nb_2.to_csv('/Users/jacquelinechou/Downloads/s2_r1/slide1_region2_kmeans_2_df.csv')

# Add these spatial neighborhood_k=2 to the anndata object
slide1_region2_tissue.obs['spatial_neighborhood_k_2'] = labels2

plt.figure(figsize=(20, 30), dpi= 300)
for color, group in slide1_region2_tissue.obs.groupby(['spatial_neighborhood_k_2']):
    idx = color[0]
    # print(idx)
    # let's subset the adata to these neighborhoods
    plt.scatter(x = group['spatial_x'], y = group['spatial_y'], s= 1, c = list(mcolors.TABLEAU_COLORS)[idx], alpha = 0.5,
                label= idx)
plt.legend(loc='upper right', markerscale = 10, fontsize = 20,
           # bbox_to_anchor = (1.25, 0.5)
)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.savefig('/Users/jacquelinechou/Downloads/s2_r1/s2_r1_test_spatial_plot_k_2.png')

k = 3
# Instantiate the k-means estimator
kmeans3 = KMeans(n_clusters=3, random_state = 16)
# Fit the model to the data
kmeans3.fit(fractional_neighbors_vector_no_nans)
# Get the cluster centroids
centroids3 = kmeans3.cluster_centers_
# Get the cluster labels for each data point
labels3 = kmeans3.labels_

# Let's also save centroids5 so we have this to refer back to
np.save('/Users/jacquelinechou/Downloads/s2_r1/s2_r1_centroids3_kmeans3.npy', centroids3)

spatial_nb_3 = pd.DataFrame(centroids3, columns = celltype_labels)

spatial_nb_3['Spatial_neighborhood'] = ['Spatial cluster 0', 'Spatial cluster 1', 'Spatial cluster 2']
# Save the dataframe as a csv file with the cell-type proportion per spatial group
spatial_nb_3.to_csv('/Users/jacquelinechou/Downloads/s2_r1/slide1_region2_kmeans_3_df.csv')

# Add these spatial neighborhood_k=2 to the anndata object
slide1_region2_tissue.obs['spatial_neighborhood_k_3'] = labels3

plt.figure(figsize=(20, 30), dpi= 300)
for color, group in slide1_region2_tissue.obs.groupby(['spatial_neighborhood_k_3']):
    idx = color[0]
    # print(idx)
    # let's subset the adata to these neighborhoods
    plt.scatter(x = group['spatial_x'], y = group['spatial_y'], s= 1, c = list(mcolors.TABLEAU_COLORS)[idx], alpha = 0.5,
                label= idx)
plt.legend(loc='upper right', markerscale = 10, fontsize = 20,
           # bbox_to_anchor = (1.25, 0.5)
)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.savefig('/Users/jacquelinechou/Downloads/s2_r1/s2_r1_test_spatial_plot_k_3.png')

k = 4
# Instantiate the k-means estimator
kmeans4 = KMeans(n_clusters=4, random_state = 16)
# Fit the model to the data
kmeans4.fit(fractional_neighbors_vector_no_nans)
# Get the cluster centroids
centroids4 = kmeans4.cluster_centers_
# Get the cluster labels for each data point
labels4 = kmeans4.labels_

# Let's also save centroids5 so we have this to refer back to
np.save('/Users/jacquelinechou/Downloads/s2_r1/s2_r1_centroids4_kmeans4.npy', centroids4)

spatial_nb_4 = pd.DataFrame(centroids4, columns = celltype_labels)

spatial_nb_4['Spatial_neighborhood'] = ['Spatial cluster 0', 'Spatial cluster 1', 'Spatial cluster 2', 'Spatial cluster 3']
# Save the dataframe as a csv file with the cell-type proportion per spatial group
spatial_nb_4.to_csv('/Users/jacquelinechou/Downloads/s2_r1/slide1_region2_kmeans_4_df.csv')

# Add these spatial neighborhood_k=2 to the anndata object
slide1_region2_tissue.obs['spatial_neighborhood_k_4'] = labels4

plt.figure(figsize=(20, 30), dpi= 300)
for color, group in slide1_region2_tissue.obs.groupby(['spatial_neighborhood_k_4']):
    idx = color[0]
    # print(idx)
    # let's subset the adata to these neighborhoods
    plt.scatter(x = group['spatial_x'], y = group['spatial_y'], s= 1, c = list(mcolors.TABLEAU_COLORS)[idx], alpha = 0.5,
                label= idx)
plt.legend(loc='upper right', markerscale = 10, fontsize = 20,
           # bbox_to_anchor = (1.25, 0.5)
)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.savefig('/Users/jacquelinechou/Downloads/s2_r1/s2_r1_test_spatial_plot_k_4.png')

# Let's run Silhouette_score from scitkit-learn to see which k is 'optimal'
from sklearn.metrics import silhouette_samples, silhouette_score
# This provides the silhouette score per data point
sample_silhouette_values = silhouette_samples(fractional_neighbors_vector_no_nans, labels4)

# Let's plot the values per data point:
# silhouette plot
y_ticks = []
y_lower = y_upper = 0

fig, ax = plt.subplots(1, 1, figsize = (15, 15))
for i, cluster in enumerate(np.unique(labels4)):
    cluster_silhouette_vals = sample_silhouette_values[labels4 == cluster]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)

    ax.barh(range(y_lower, y_upper),
               cluster_silhouette_vals, height=1);
    ax.text(-0.03, (y_lower + y_upper) / 2, str(i))
    y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score
    avg_score = np.mean(sample_silhouette_values)
    ax.axvline(avg_score, linestyle='--',
                  linewidth=2, color='green')
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.set_xlabel('Silhouette coefficient values')
    ax.set_ylabel('Cluster labels')
    ax.set_title('Silhouette plot for the various clusters')
plt.savefig('/Users/jacquelinechou/Downloads/s2_r1/s2_r1_silhouette_plot_k_4.png')

slide1_region2_tissue.write_h5ad('/Users/jacquelinechou/Downloads/240718_s1_r2_post_kmeans_tissue.h5ad')