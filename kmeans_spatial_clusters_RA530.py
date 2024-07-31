import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.neighbors import KDTree

concat_ad = sc.read_h5ad('/Users/jacquelinechou/Downloads/240522_cohort_all_ctd.h5ad')

# Let's go ahead and rename some of the identified clusters from Leiden_res = 0.2

concat_ad.obs["cell_type_tier1"] = concat_ad.obs["leiden_res_0_3"].map(
    {
        "0": "Myeloid cells",
        "1": "Sublining fibroblasts",
        "2": "T-cells",
        "3": "Lining fibroblasts",
        "4": "Endothelial cells",
        "5": "Plasma cells",
        "6": "Non-plasma B-cells",
        "7": "Mast cells/ adipocytes",
        "8": "Non-plasma cells"
    }
)

# Let's just look at slide2_region4 (T/B aggregate heavy, good to test out spatial niche test)
slide2_region4_tissue = concat_ad[concat_ad.obs['tissue'] == 'slide2_r4']

# Create an adjacency matrix for all 141K cells with a radius = 100 um
def generate_neighbor_graph(single_cell_matrix, distance_threshold=100):
    ## Generates the GRAPH for all cells in data within 100 microns of each other
    kdt = KDTree(single_cell_matrix.obsm['spatial'], metric='euclidean')

    # get the nearest 100 meighbors using kdTree
    distances_matrix, neighbor_indices = kdt.query(single_cell_matrix.obsm['spatial'], k=151, return_distance=True)
    distances_matrix, neighbor_indices = distances_matrix[:, 1:], neighbor_indices[:, 1:]  # this removes the first column which is itself
    distances_mask = distances_matrix < distance_threshold
    return distances_matrix, neighbor_indices, distances_mask

distances_matrix, neighbor_graph, distances_mask = generate_neighbor_graph(slide2_region4_tissue)

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
positions = {i: tuple(coord) for i, coord in enumerate(slide2_region4_tissue.obsm['spatial'])}
for node, pos in positions.items():
    G.add_node(node, pos=pos)

# Add edges based on distances_mask and neighbor_indices
edge_indices = np.where(distances_mask == 1)
edges = [(i, neighbor_graph[i, k]) for i, k in zip(*edge_indices)]
G.add_edges_from(edges)

# Plot the neighbor connectivity: can't really discern the T/B aggregates, too many other cells nearby
plt.figure(figsize=(30*3, 30*3), dpi = 300)
nx.draw(G, positions, with_labels=False, node_color='lightblue', node_size=1, font_size=10, font_weight='bold', edge_color='gray')
plt.title("Spatial Network: Slide2 Region 4"); plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()
plt.savefig('/Users/jacquelinechou/Downloads/s2_r4_spatial_network_90x90_300dpi.png');

# Get the labels of the neighbors and calculate the vector of cell-types per cell
# Run k-means clustering on those vectors of cell-types to get structures
# new_indices = np.arange(0, slide2_region4_tissue.obs.shape[0], 1).astype('str')
# s1_r4_df_reindexed = slide2_region4_tissue
# s1_r4_df_reindexed.obs = slide2_region4_tissue.obs.set_index(new_indices)

# Let's get the cell-type labels per cell using neighbor graph
subset_list = [slide2_region4_tissue.obs['cell_type_tier1'].iloc[neighbors].values for neighbors in filtered_neighbors]

# # Let's fix the cell-type that's named 'Non-plasma cells' to 'Non-plasma B-cells'
# s1_r4_df_reindexed.obs["cell_type_tier1_fixed"] = s1_r4_df_reindexed.obs["cell_type_tier1"].map(
#     {   "Myeloid cells": "Myeloid cells",
#         "Sublining fibroblasts": "Sublining fibroblasts",
#         "T-cells": "T-cells",
#         "Lining fibroblasts": "Lining fibroblasts",
#         "Endothelial cells": "Endothelial cells",
#         "Plasma cells": "Plasma cells",
#         "Non-plasma B-cells": "Non-plasma B-cells",
#         "Mast cells/ adipocytes": "Mast cells/ adipocytes",
#         "Non-plasma cells": 'Non-plasma B-cells'
#     }
# )

celltype_labels = slide2_region4_tissue.obs['cell_type_tier1_fixed'].unique()

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
kmeans = KMeans(n_clusters=2)
# Fit the model to the data
kmeans.fit(fractional_neighbors_vector_no_nans)
# Get the cluster centroids
centroids = kmeans.cluster_centers_
# Get the cluster labels for each data point
labels = kmeans.labels_

# Add these spatial neighborhood_k=2 to the anndata object
slide2_region4_tissue.obs['spatial_neighborhood_k_2'] = labels

# Let's plot what the 'spatial_neighborhood_k_2' looks like
plt.figure(figsize= (60, 60), dpi = 100)
plt.scatter(x=slide2_region4_tissue[slide2_region4_tissue.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
            y=slide2_region4_tissue[slide2_region4_tissue.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
            c = 'yellow', s = 1)
plt.legend([''])
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()
plt.savefig('/Users/jacquelinechou/Downloads/s2_r4/s2_r4_neighbor_k_5_spatial_plot.png')

# Let's include the legend to the spatial plot of spatial neighborhoods
kmeans5 = KMeans(n_clusters=5)
# Fit the model to the data
kmeans5.fit(fractional_neighbors_vector_no_nans)
# Get the cluster centroids
centroids5 = kmeans5.cluster_centers_
# Get the cluster labels for each data point
labels5 = kmeans5.labels_

# Add these spatial neighborhood_k=2 to the anndata object
slide2_region4_tissue.obs['spatial_neighborhood_k_5'] = labels5

from matplotlib import colors as mcolors
slide2_region4_tissue.obs['spatial_x'] = slide2_region4_tissue.obsm['spatial'][:, 0]
slide2_region4_tissue.obs['spatial_y'] = slide2_region4_tissue.obsm['spatial'][:, 1]

plt.figure(figsize=(20/2, 30/2), dpi= 300)
for color, group in slide2_region4_tissue.obs.groupby(['spatial_neighborhood_k_5']):
    idx = color[0]
    # print(idx)
    # let's subset the adata to these neighborhoods
    plt.scatter(x = group['spatial_x'], y = group['spatial_y'], s= 1, c = list(mcolors.TABLEAU_COLORS)[idx], alpha = 0.5,
                label= idx)
plt.legend(loc='upper right', markerscale = 10, fontsize = 20, bbox_to_anchor = (1.25, 0.5)); plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
# legend.legendHandles[0]._legmarker.set_markersize(10)
# legend.legendHandles[0]._legmarker.set_alpha(1)
plt.savefig('/Users/jacquelinechou/Downloads/s2_r4/s2_r4_test_spatial_plot_k_5.png')

# Let's make the stacked barplot of the spatial clusters identified using kmeans
spatial_neighborhoods_df = pd.DataFrame(centroids5, columns = ['T-cells', 'Sublining fibroblasts',
                                                               'Endothelial cells', 'Mast cells/ adipocytes', 'Myeloid cells',
                                                               'Plasma cells', 'Non-plasma B-cells', 'Lining fibroblasts', 'nan'])

spatial_neighborhoods_df['Spatial_neighborhood'] = ['Spatial cluster 0', 'Spatial cluster 1', 'Spatial cluster 2',
                                                    'Spatial cluster 3', 'Spatial cluster 4']

# Stacked barplot
spatial_neighborhoods_df.plot(x = 'Spatial_neighborhood', kind = 'bar', stacked=True)

# Let's create a new column in the anndata wherein we create a boolean mask for whether sublining fibroblasts are within spatial neighborhood, k=2
slide2_region4_tissue.obs['fib_k=2_niche'] = np.where((slide2_region4_tissue.obs['cell_type_tier1_fixed'] == 'Sublining fibroblasts') &
                                                   (slide2_region4_tissue.obs['spatial_neighborhood_k_5'] == 3), '1', '0')

# Let's run differential gene expression analysis using scanpy
sc.tl.rank_genes_groups(slide2_region4_tissue, groupby = 'fib_k=2_niche', method = 'wilcoxon', key_added = 'fib_k=2_niche_degs')

sc.pl.rank_genes_groups_dotplot(
    s1_r4_df_reindexed, groupby="fib_k=2_niche", standard_scale="var", n_genes=15, key = 'fib_k=2_niche_degs',
    save = 's1_r4_fibs_leiden_tb_spatial_test.png')

# So it looks like the DEGs for sublining fibroblasts inside T/B aggregates is also
# expressed in sublining fiborblasts elsewhere, but not as highly expressed?

# Let's look at CXCL12 expression in this tissue
plt.figure(figsize = (40, 20), dpi = 200)
plt.scatter(x = s2_r4_df_reindexed.obsm['spatial'][:, 0], y = s2_r4_df_reindexed.obsm['spatial'][:, 1], s =1 ,
            c = s2_r4_df_reindexed.X[:, s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'CXCL12'].toarray());
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.colorbar()
# plt.savefig('/Users/jacquelinechou/s2_r4_cxcl12_expression_all_cells.png')
plt.show()

# Let's plot CXCL12 expression in sublining fibroblasts only
plt.figure(figsize = (40*1.5, 20*1.5), dpi = 300)
plt.scatter(x = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 0]),
                y = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 1]), s =1 ,
            c = np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
            s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'CXCL12']))
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('CXCL12 expression in sublining fibroblasts only (S2_R4)')
plt.colorbar(shrink = 0.25)
# plt.savefig('/Users/jacquelinechou/s2_r4_cxcl12_expression_all_sublining_fibroblasts.png')
plt.show()

# Let's plot the spatial_neighborhoods and the CXCL12 expression in sublining fibroblasts overlayed
plt.figure(figsize = (40/2, 20/2), dpi = 300)
plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
            y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
            c = 'lightskyblue', s = 1)
plt.scatter(x = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 0]),
                y = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 1]), s =1 , cmap =
                                                                                                                                       'inferno_r',
            c = np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
            s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'CXCL12']))
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('CXCL12 expression in sublining fibroblasts only  \n spatial_neighborhood = 2 (S2_R4)')
plt.colorbar(shrink = 0.5)
plt.savefig('/Users/jacquelinechou/Downloads/test_s2_r4_cxcl12_expression_all_sublining_fibroblasts.png')
plt.show()

plt.figure(figsize = (40/2, 20/2), dpi = 300)
plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
            y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
            c = 'lightskyblue', s = 1)
plt.scatter(x = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 0]),
                y = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 1]), s =1 , cmap = 'inferno_r',
            c = np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
            s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'CCL2']))
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('CCL2 expression in sublining fibroblasts only  \n spatial_neighborhood = 2 (S2_R4)')
plt.colorbar(shrink = 0.5)
plt.savefig('/Users/jacquelinechou/Downloads/test_s2_r4_ccl2_expression_all_sublining_fibroblasts.png')
plt.show()

plt.figure(figsize = (40/2, 20/2), dpi = 300)
plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
            y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
            c = 'lightskyblue', s = 1)
plt.scatter(x = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 0]),
                y = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 1]), s =1 , cmap =
                                                                                                                                       'inferno_r',
            c = np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
            s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'PTGDS']))
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('PTGDS expression in sublining fibroblasts only  \n spatial_neighborhood = 2 (S2_R4)')
plt.colorbar(shrink = 0.5)
plt.savefig('/Users/jacquelinechou/Downloads/test_s2_r4_ptgds_expression_all_sublining_fibroblasts.png')
plt.show()

top_fibs_degs_spatial_k2 = ['CXCL12', 'PTGDS', 'VCAN', 'TNC', 'FBN1', 'CTSK']
for gene in np.arange(len(top_fibs_degs_spatial_k2)):
    gene_of_interest = top_fibs_degs_spatial_k2[gene]
    plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
    plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
                y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
                c='lightskyblue', s=1)
    plt.scatter(x=np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 0]),
                y=np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 1]),
                s=1, cmap= 'inferno_r',
                c=np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
                s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == gene_of_interest]))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title(f'{gene_of_interest} expression in sublining fibroblasts only  \n spatial_neighborhood = 2 (S2_R4)')
    plt.colorbar(shrink=0.5)
    plt.savefig(f'/Users/jacquelinechou/Downloads/test_s2_r4_{gene_of_interest}_expression_all_sublining_fibroblasts.png')
    plt.show()

# Let's plot all top 15 genes (add all raw counts)
top_fibs_15_degs_spatial_k2 = ['CXCL12', 'PTGDS', 'VCAN', 'TNC', 'FBN1', 'CTSK', 'GEM', 'DPT', 'SPIB', 'CCL19',
                               'IGFBP3', 'FAS', 'CTNNB1', 'VEGFA', 'CCL2']
leiden_condition = s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'
genes_condition = s2_r4_df_reindexed.var_names.isin(top_fibs_15_degs_spatial_k2)
top15_gene_count = s2_r4_df_reindexed.layers['counts'][leiden_condition][:, genes_condition].sum(axis = 1)
# Extract the spatial data for the specific condition
spatial_data = s2_r4_df_reindexed[leiden_condition].obsm['spatial']

plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
# Plot the spatial neighborhood, k = 2
plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
            y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
            c='lightskyblue', s=1)
plt.scatter(x=np.array(s2_r4_df_reindexed[leiden_condition].obsm['spatial'][:, 0]),
            y=np.array(s2_r4_df_reindexed[leiden_condition].obsm['spatial'][:, 1]),
            s=1, cmap= 'inferno_r', c= np.array(top15_gene_count).reshape(-1))
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('Top 15 DEGs expression in sublining fibroblasts only  \n spatial_neighborhood = 2 (S2_R4)')
plt.colorbar(shrink=0.5)
# plt.savefig(f'/Users/jacquelinechou/Downloads/test_s2_r4_top_15_gene_expression_all_sublining_fibroblasts.png')
plt.show()

# Let's create a new column in the anndata wherein we create a boolean mask for whether sublining fibroblasts are within spatial neighborhood, k=2
s2_r4_df_reindexed.obs['tcell_k=2_niche'] = np.where((s2_r4_df_reindexed.obs['leiden_res_0_3'] == '2') &
                                                   (s2_r4_df_reindexed.obs['spatial_neighborhood_k_5'] == 2), '1', '0')

# Let's run differential gene expression analysis using scanpy
sc.tl.rank_genes_groups(s2_r4_df_reindexed, groupby = 'tcell_k=2_niche', method = 'wilcoxon', key_added = 'tcell_k=2_niche_degs')

sc.pl.rank_genes_groups_dotplot(
    s2_r4_df_reindexed, groupby="tcell_k=2_niche", standard_scale="var", n_genes=25, key = 'tcell_k=2_niche_degs',
    save = '240606_tcell_leiden_tb_spatial_test.png')

# How are the non-plasma B-cells in these aggregates different from other non-plasma B-cells?
s2_r4_df_reindexed.obs['nonplasma_bcell_k=2_niche'] = np.where((s2_r4_df_reindexed.obs['cell_type_tier1_fixed'] == 'Non-plasma B-cells') &
                                                   (s2_r4_df_reindexed.obs['spatial_neighborhood_k_5'] == 2), '1', '0')
# Let's run differential gene expression analysis using scanpy
sc.tl.rank_genes_groups(s2_r4_df_reindexed, groupby = 'nonplasma_bcell_k=2_niche', method = 'wilcoxon', key_added = 'nonplasma_bcell_k=2_niche_degs')

sc.pl.rank_genes_groups_dotplot(
    s2_r4_df_reindexed, groupby="nonplasma_bcell_k=2_niche", standard_scale="var", n_genes=15, key = 'nonplasma_bcell_k=2_niche_degs',
    save = '240606_nonplasma_bcell_leiden_tb_spatial_test.png')

# Instead of just visualizing a few DEGs, we can get a dataframe with the DEGs
df = sc.get.rank_genes_groups_df(s2_r4_df_reindexed, group='1', key='tcell_k=2_niche_degs')

with plt.rc_context({"figure.figsize": (4.5, 3)}):
    sc.pl.violin(s2_r4_df_reindexed, ["TRAC", "MS4A1", "PTPRC", "CXCL13"], groupby="tcell_k=2_niche")

# Let's look at the spatial localization of these DEGs wrt their cell-type location
top_tcells_degs_spatial_k2 = list(df['names'][0:10])
for gene in np.arange(len(top_tcells_degs_spatial_k2)):
    gene_of_interest = top_tcells_degs_spatial_k2[gene]
    print(gene_of_interest)
    plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
    plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
                y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
                c='lightskyblue', s=1)
    plt.scatter(x=np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['cell_type_tier1_fixed'] == 'T-cells'].obsm['spatial'][:, 0]),
                y=np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['cell_type_tier1_fixed'] == 'T-cells'].obsm['spatial'][:, 1]),
                s=1, cmap= 'inferno_r', alpha = 0.5,
                c=np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['cell_type_tier1_fixed'] == 'T-cells'),
                s2_r4_df_reindexed[s2_r4_df_reindexed.obs['cell_type_tier1_fixed'] == 'T-cells'].var_names == gene_of_interest]))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title(f'{gene_of_interest} expression in T-cells only  \n spatial_neighborhood = 2 (S2_R4)')
    plt.colorbar(shrink=0.5)
    plt.savefig(f'/Users/jacquelinechou/Downloads/s2_r4_{gene_of_interest}_expression_all_tcells.png')

top_nonplasma_cells_degs_spatial_k2 = list(df_bcells['names'][0:10])
for gene in np.arange(len(top_nonplasma_cells_degs_spatial_k2)):
    gene_of_interest = top_nonplasma_cells_degs_spatial_k2[gene]
    print(gene_of_interest)
    plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
    plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
                y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
                c='lightskyblue', s=1)
    plt.scatter(x=np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['cell_type_tier1_fixed'] == 'Non-plasma B-cells'].obsm['spatial'][:, 0]),
                y=np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['cell_type_tier1_fixed'] == 'Non-plasma B-cells'].obsm['spatial'][:, 1]),
                s=1, cmap= 'inferno_r', alpha = 0.5,
                c=np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['cell_type_tier1_fixed'] == 'Non-plasma B-cells'),
                s2_r4_df_reindexed[s2_r4_df_reindexed.obs['cell_type_tier1_fixed'] == 'Non-plasma B-cells'].var_names == gene_of_interest]))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title(f'{gene_of_interest} expression in Non-plasma B-cells only  \n spatial_neighborhood = 2 (S2_R4)')
    plt.colorbar(shrink=0.5)
    plt.savefig(f'/Users/jacquelinechou/Downloads/s2_r4_{gene_of_interest}_expression_all_non_plasma_bcells.png')

# Let's tweak which fibroblasts we look at; rather than looking exclusively at just fibroblasts in spatial_group =2
# Let's look at proximal fibroblasts (within say, 20 microns of spatial_group =2) and compare to distal fibroblasts
# And see the difference in DEGs

new_indices = np.arange(0, slide2_region4_tissue.obs.shape[0], 1).astype('str')
slide2_region4_tissue.obs.set_index(new_indices, inplace = True)

from scipy.spatial import KDTree
# Get the spatial coordinates of the cells in T/B aggregates (spatial group = 2)
tb_spatial_loc = slide2_region4_tissue.obsm['spatial'][slide2_region4_tissue.obs['spatial_neighborhood_k_5'] == 1, :]
# Get the spatial coordinates of the sublining fibroblasts nearest the T/B aggregates
# fibroblasts = slide2_region4_tissue.obsm['spatial'][slide2_region4_tissue.obs['celltype_tier1_merged'] == 'Sublining fibroblasts', :]

# Rather than jst looking at sublining fibroblasts, let's find all proximal cells (w/i 30 microns)
all_cells = slide2_region4_tissue.obsm['spatial']
# Build the KD-tree using the spatial locations of the TB aggregate
kdtree = KDTree(all_cells)

distances_threshold = 30

# neighbors_within_threshold:
proximal_cells_tb = kdtree.query_ball_point(tb_spatial_loc, r = distances_threshold)
proximal_cells_tb_indices = np.unique(np.concatenate(proximal_cells_tb))

slide2_region4_tissue.obs['proximal_to_tb'] = 0
slide2_region4_tissue.obs.iloc[proximal_cells_tb_indices, slide2_region4_tissue.obs.columns.get_loc('proximal_to_tb')] = 1

plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
# Plot the spatial neighborhood, T/B aggregates
plt.scatter(x = tb_spatial_loc[:, 0], y = tb_spatial_loc[:, 1], s = 1)
# filter adata to just the proximal cells
proximal_x_coords = slide2_region4_tissue[slide2_region4_tissue.obs['proximal_to_tb'] == 1].obsm['spatial'][:, 0]
proximal_y_coords = slide2_region4_tissue[slide2_region4_tissue.obs['proximal_to_tb'] == 1].obsm['spatial'][:, 1]
plt.scatter(x = proximal_x_coords, y = proximal_y_coords, s = 1)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.savefig('/Users/jacquelinechou/Downloads/s2_r4_tb_kdtree_30_microns.png')
#
# # Let's just use distance = 30 microns as proximal, and any cells further than 3 microns are considered distal cells
# slide2_region4_tissue.obs['proximal_sublining_fibs_30microns'] = (distances < distances_threshold)

# Let's reindex the .obs dataframe and create column for proximal_cells

# # Then conditioned on the cell-type and proximal_cells column, we can create a new column for cell-type specific proximal cells
# slide2_region4_tissue.obs['proximal_to_tb_sublining_fibs'] = np.where((slide2_region4_tissue.obs['celltype_tier1_merged'] == 'Sublining fibroblasts') &
#                                                    (slide2_region4_tissue.obs['proximal_cells_tb_30microns'] == '1'), '1', '0')
#
# # Let's run differential gene expression analysis on the proximal v distal fibroblasts
# sc.tl.rank_genes_groups(slide2_region4_tissue, groupby = 'proximal_to_tb_sublining_fibs', method = 'wilcoxon',
#                         key_added = 'proximal_sublining_fibs_degs')
#
# sc.pl.rank_genes_groups_dotplot(
#     slide2_region4_tissue, groupby="proximal_to_tb_sublining_fibs", standard_scale="var", n_genes=15, key = 'proximal_sublining_fibs_degs',
#     save = 's2_r4_sublining_fibs_30_microns_tb_agg_degs_dotplot.png')

# Let's create a column with celltype names and whether it's proximal (1) or distal to TB spatial niche
slide2_region4_tissue.obs['proximal_to_tb'] = slide2_region4_tissue.obs['proximal_to_tb'].astype('str')
slide2_region4_tissue.obs['celltype_tier1_proximal_tb'] = slide2_region4_tissue.obs['celltype_tier1_merged'].str.cat(slide2_region4_tissue.obs['proximal_to_tb'])

slide2_region4_tissue.write_h5ad('/Users/jacquelinechou/Downloads/240701_s2_r4_updated.h5ad')

sc.tl.rank_genes_groups(slide2_region4_tissue, groupby = 'celltype_tier1_proximal_tb', groups = ['Sublining fibroblasts1'],
                        reference = 'Sublining fibroblasts0', method = 'wilcoxon', key_added = 'sublining_fibs_1_ref_0_degs')

sc.pl.rank_genes_groups_dotplot(
    slide2_region4_tissue, groupby = 'celltype_tier1_proximal_tb', standard_scale="var", n_genes=15, key = 'sublining_fibs_1_ref_0_degs',
    save = 's2_r4_sublining_fibs_in_tb_niche_vs_out_dotplot.png')

# To just plot the Sublining_fibs in the spatial niche and outside of the spatial niche in a violin plot, I have to subset the adata
mask = slide2_region4_tissue.obs['celltype_tier1_proximal_tb'].isin(['Sublining fibroblasts1', 'Sublining fibroblasts0'])

# violin plot
sc.pl.violin(slide2_region4_tissue[mask], keys= ['PTGDS', 'TNC', 'CXCL12'], groupby = 'celltype_tier1_proximal_tb',
             save = 'upregulated_genes_sublining_fibs_tb_niche_vs_out.png')

# Let's look at DEGs for myeloid, T-cells, B-cells and lining fibroblasts
celltypes_interest = ['Myeloid cells', 'T-cells', 'Non-plasma B-cells', 'Lining fibroblasts']

for celltype in celltypes_interest:
    sc.tl.rank_genes_groups(slide2_region4_tissue, groupby = 'celltype_tier1_proximal_tb', groups = [f'{celltype}1'],
                        reference = f'{celltype}0', method = 'wilcoxon', key_added = f'{celltype}_1_ref_0_degs')

    sc.pl.rank_genes_groups_dotplot(
        slide2_region4_tissue, groupby = 'celltype_tier1_proximal_tb', standard_scale="var", n_genes=15, key = f'{celltype}_1_ref_0_degs',
        save = f's2_r4_{celltype}_in_tb_niche_vs_out_dotplot.png')
    # Create mask for cell-types of interest for plotting [0 v. 1 per celltype]
    mask = slide2_region4_tissue.obs['celltype_tier1_proximal_tb'].isin([f'{celltype}1', f'{celltype}0'])
    top_5_genes = pd.DataFrame(slide2_region4_tissue.uns[f'{celltype}_1_ref_0_degs']['names']).head(5)
    genes_to_plot = top_5_genes.values.flatten()
    # Let's make the violin plots for each celltype:
    sc.pl.violin(slide2_region4_tissue[mask], keys = genes_to_plot, groupby = 'celltype_tier1_proximal_tb', save = f's2_r4_{celltype}_tb_1_0.png')

# Let's get the CSV file of lining fibroblasts' DEGs:
lining_fib_tb_degs = sc.get.rank_genes_groups_df(slide2_region4_tissue, key = 'Lining fibroblasts_1_ref_0_degs',
                                                 group = 'Lining fibroblasts1', pval_cutoff= 0.05)

lining_fib_tb_degs.to_csv('/Users/jacquelinechou/Downloads/s2_r4_lining_fib_tb_degs.csv')

# Let's resave the h5ad file
slide2_region4_tissue.write_h5ad('/Users/jacquelinechou/Downloads/240701_s2_r4_updated.h5ad')

# Let's plot the spatial_niche (T/B) with the different cell-types highlighted on it
plt.figure(figsize = (40/2, 20/2), dpi = 200)
# Plot the spatial niche as sky blue
plt.scatter(x=slide2_region4_tissue[slide2_region4_tissue.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
            y=slide2_region4_tissue[slide2_region4_tissue.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
            c = 'lightskyblue', s = 1)
# Plot all myeloid cells across the tissue
plt.scatter(x = np.array(slide2_region4_tissue[slide2_region4_tissue.obs['celltype_tier1_merged'] == 'Myeloid cells'].obsm['spatial'][:, 0]),
                y = np.array(slide2_region4_tissue[slide2_region4_tissue.obs['celltype_tier1_merged'] == 'Myeloid cells'].obsm['spatial'][:, 1]), s =1,
                c = 'red',
            #cmap = 'inferno_r',
            # c = np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
            # s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'CCL2'])
            )
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('Presence of Myeloid cells  \n spatial_neighborhood = 2 (S2_R4)')
# plt.colorbar(shrink = 0.5)
plt.savefig('/Users/jacquelinechou/Downloads/s2_r4_tb_agg_all_myeloid_cells_overlayed.png')
plt.show()

# Let's overlay the H/E image with the spatial niche
# Let's overlay the cell-type over the spatial niche

import tifffile
import pandas as pd
import cv2

s2_r4_histology = tifffile.imread('/Users/jacquelinechou/Documents/Histology_post_Xenium_split_tissues_ome_tiff/S2 - R4.ome.tiff')
matrix = pd.read_csv('/Users/jacquelinechou/Downloads/S2_R4_histology_alignment_files/s2_r4_matrix.csv', header = None)

matrix = np.float32(matrix)

twod_matrix = matrix[0:2, ]
image_xenium = tifffile.imread('/Users/jacquelinechou/10X_Xenium_dataset_231024/Slide_2/output-XETG00065__0005398__Region_4__20231018__185547/morphology_mip.ome.tif')

transformed_image = cv2.warpAffine(s2_r4_histology, twod_matrix, (image_xenium.shape[0], image_xenium.shape[1]))

# The H/E image is larger than cv2 can handle, so I need to tile it and then merge after applying the transformation
import cv2
import numpy as np

def split_image_with_overlap(image, tile_size, overlap):
    """Splits the image into smaller tiles of tile_size x tile_size with overlap."""
    tiles = []
    h, w = image.shape[:2]
    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, y, x))
    return tiles

def merge_tiles_with_overlap(tiles, image_shape, tile_size, overlap):
    """Reassembles the tiles into the original image shape with blending."""
    h, w = image_shape[:2]
    image = np.zeros(image_shape, dtype=tiles[0][0].dtype)
    # print(image.dtype)
    # weight = np.zeros(image_shape[:2], dtype=np.uint16)
    # print(weight.dtype)
    for tile, y, x in tiles:
        print(f'x: {x}, y: {y}')
        y1, y2 = max(0, y), min(h, y + tile.shape[0])
        print(y1, y2)
        x1, x2 = max(0, x), min(w, x + tile.shape[1])
        print(x1, x2)
        image[y1:y2, x1:x2] += tile[:(y2 - y1), :(x2 - x1)]
        # weight[y1:y2, x1:x2] += 1

    # Normalize the image by dividing by the weight
    # image = image / np.maximum(weight[:, :, None], 1)  # Avoid division by zero

    return image

def apply_warp_affine_to_tiles_with_overlap(image, tile_size, overlap, M):
    """Applies warpAffine to each tile of the image with overlap and blending."""
    tiles = split_image_with_overlap(image, tile_size, overlap)
    # print(len(tiles))
    transformed_tiles = []
    for tile, y, x in tiles:
        # print(f'tile: {tile}')
        # Adjust the transformation matrix for each tile
        # offset_matrix = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]])
        # print(f'offset_matrix: {offset_matrix}')
        # M_tile = np.dot(offset_matrix, np.vstack((M, [0, 0, 1])))[:2]
        transformed_tile = cv2.warpAffine(tile, M_tile, (tile.shape[1], tile.shape[0]))
        transformed_tiles.append((transformed_tile, y, x))
    # return transformed_tiles
    return merge_tiles_with_overlap(transformed_tiles, image.shape, tile_size, overlap)

# Set the maximum tile size to SHRT_MAX and define the overlap size
SHRT_MAX = 3270
max_tile_size = SHRT_MAX
overlap_size = 10  # Define the overlap size in pixels

test = apply_warp_affine_to_tiles_with_overlap(s2_r4_histology, tile_size= 3270, overlap = 10, M = twod_matrix)

# My above code didn't work; used Big Warp from Fiji to align histology image to Xenium DAPI stain manually (had 300+ landmarks)
aligned_histology = tifffile.imread('/Users/jacquelinechou/Downloads/240630_s2_r4_manual_aligned_histology_refined.tiff')
aligned_histology_t = np.transpose(aligned_histology, (1, 2, 0))
# Had to use the next-level resolution for DAPI morphology image
plt.figure(figsize = (20, 10), dpi = 300)
plt.imshow(aligned_histology_t)
# Plot the spatial neighborhood of interest
image_scaling_factor = 0.4250
plt.scatter(x = slide2_region4_tissue[slide2_region4_tissue.obs['spatial_neighborhood_k_5'] == 1].obsm['spatial'][:, 0] * image_scaling_factor,
        y = slide2_region4_tissue[slide2_region4_tissue.obs['spatial_neighborhood_k_5'] == 1].obsm['spatial'][:, 1] * image_scaling_factor,
            c = 'lightskyblue', s = 1, facecolors = 'none')
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('Spatial niche 1 in S2_R4')
plt.savefig('/Users/jacquelinechou/Downloads/test_overlay.png')
plt.show()

# Include the scale bar
import matplotlib.patches as patches

scale_bar_length = 1000  # in microns (µm)
pixel_size = 0.425  # size of a pixel in microns (µm)
scale_bar_pixel_length = scale_bar_length / pixel_size  # scale bar length in pixels

fig, ax = plt.subplots(figsize = (20, 10), dpi = 1000)
ax.imshow(aligned_histology_t)
# Add the scale bar
x_position = aligned_histology_t.shape[1] - 50 - scale_bar_pixel_length
y_position = aligned_histology_t.shape[0] - 50 - 10

rect = patches.Rectangle((x_position, y_position), scale_bar_pixel_length, 10,
                         linewidth=1, edgecolor='black', facecolor='black')
ax.add_patch(rect)

# Add the scale bar text
ax.text(x_position + scale_bar_pixel_length / 2, y_position,
        f'{scale_bar_length} µm', color='black', fontsize=12, ha='center')
image_scaling_factor = 0.4250
plt.scatter(x = slide2_region4_tissue[slide2_region4_tissue.obs['spatial_neighborhood_k_5'] == 1].obsm['spatial'][:, 0] / image_scaling_factor,
        y = slide2_region4_tissue[slide2_region4_tissue.obs['spatial_neighborhood_k_5'] == 1].obsm['spatial'][:, 1] / image_scaling_factor,
            c = 'lightskyblue', s = 1, facecolors = 'none', alpha = 0.5)
ax.set_aspect('equal')
ax.title('Spatial niche 1 in S2_R4')
plt.savefig('/Users/jacquelinechou/Downloads/test_overlay.png')
plt.show()

# Let's show a specific T/B aggregate:
x_start = 16_000
x_end = 19_000
y_start = 9_000
y_end = 10_600

image_scaling_factor = 0.4250
# Subset the adata to just cells of interest
specific_roi = slide2_region4_tissue[(slide2_region4_tissue.obsm['spatial'][:, 0] > 16_000 * image_scaling_factor) &
                                     (slide2_region4_tissue.obsm['spatial'][:, 0] < 19_000 * image_scaling_factor)
                                     & (slide2_region4_tissue.obsm['spatial'][:, 1] > 9_000 * image_scaling_factor)
                                     & (slide2_region4_tissue.obsm['spatial'][:, 1] < 10_600 * image_scaling_factor)]
plt.figure(figsize = (20, 10), dpi = 200)
plt.imshow(aligned_histology_t[y_start:y_end, x_start:x_end])
plt.scatter(x = specific_roi[specific_roi.obs['spatial_neighborhood_k_5'] == 2].obsm['spatial'][:, 0] / image_scaling_factor,
        y = specific_roi[specific_roi.obs['spatial_neighborhood_k_5'] == 2].obsm['spatial'][:, 1] / image_scaling_factor,
            c = 'lightskyblue', s = 1, facecolors = 'none', alpha = 0.1)
plt.savefig('/Users/jacquelinechou/Downloads/cropped_s2_r4_histology_overlay_spatial_niche.png')