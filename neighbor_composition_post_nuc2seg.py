import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

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

masked_indices = np.where(mask, neighbor_indices, -1)

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
new_indices = np.arange(0, slide2_region4_tissue.obs.shape[0], 1).astype('str')
s2_r4_df_reindexed = slide2_region4_tissue
s2_r4_df_reindexed.obs = slide2_region4_tissue.obs.set_index(new_indices)

# Let's get the cell-type labels per cell using neighbor graph
subset_list = [s2_r4_df_reindexed.obs['cell_type_tier1'].iloc[neighbors].values for neighbors in filtered_neighbors]



# Let's fix the cell-type that's named 'Non-plasma cells' to 'Non-plasma B-cells'
s2_r4_df_reindexed.obs["cell_type_tier1_fixed"] = s2_r4_df_reindexed.obs["cell_type_tier1"].map(
    {   "Myeloid cells": "Myeloid cells",
        "Sublining fibroblasts": "Sublining fibroblasts",
        "T-cells": "T-cells",
        "Lining fibroblasts": "Lining fibroblasts",
        "Endothelial cells": "Endothelial cells",
        "Plasma cells": "Plasma cells",
        "Non-plasma B-cells": "Non-plasma B-cells",
        "Mast cells/ adipocytes": "Mast cells/ adipocytes",
        "Non-plasma cells": 'Non-plasma B-cells'
    }
)

celltype_labels = s2_r4_df_reindexed.obs['cell_type_tier1_fixed'].unique()

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
s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] = labels

# Let's plot what the 'spatial_neighborhood_k_2' looks like
plt.figure(figsize= (60, 60), dpi = 100)
plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
            y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
            c = 'yellow', s = 1)
plt.legend([''])
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()
plt.savefig('/Users/jacquelinechou/Downloads/s2_r4_neighbor_k_5_spatial_plot.png')

# Let's include the legend to the spatial plot of spatial neighborhoods

from matplotlib import colors as mcolors
s2_r4_df_reindexed.obs['spatial_x'] = s2_r4_df_reindexed.obsm['spatial'][:, 0]
s2_r4_df_reindexed.obs['spatial_y'] = s2_r4_df_reindexed.obsm['spatial'][:, 1]

plt.figure(figsize=(40, 20), dpi= 200)
for color, group in s2_r4_df_reindexed.obs.groupby(['spatial_neighborhood_k_5']):
    idx = color[0]
    # print(idx)
    # let's subset the adata to these neighborhoods
    plt.scatter(x = group['spatial_x'], y = group['spatial_y'], s= 1, c = list(mcolors.TABLEAU_COLORS)[idx], alpha = 0.5,
                label= idx)
plt.legend(loc='upper right', markerscale = 5, fontsize = 20); plt.gca().invert_yaxis()
# legend.legendHandles[0]._legmarker.set_markersize(10)
# legend.legendHandles[0]._legmarker.set_alpha(1)
plt.savefig('/Users/jacquelinechou/Downloads/test_spatial_plot_k_5.png')

# Let's make the stacked barplot of the spatial clusters identified using kmeans
spatial_neighborhoods_df = pd.DataFrame(centroids5, columns = [T-cells', 'Sublining fibroblasts',
                                                               'Endothelial cells', 'Mast cells/ adipocytes', 'Myeloid cells',
                                                               'Plasma cells', 'Non-plasma B-cells', 'Lining fibroblasts', 'nan'])

spatial_neighborhoods_df['Spatial_neighborhood'] = ['Spatial cluster 0', 'Spatial cluster 1', 'Spatial cluster 2',
                                                    'Spatial cluster 3', 'Spatial cluster 4']

# Let's create a new column in the anndata wherein we create a boolean mask for whether sublining fibroblasts are within spatial neighborhood, k=2
s2_r4_df_reindexed.obs['fib_k=2_niche'] = np.where((s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1') &
                                                   (s2_r4_df_reindexed.obs['spatial_neighborhood_k_5'] == 2), '1', '0')

# Let's run differential gene expression analysis using scanpy
sc.tl.rank_genes_groups(s2_r4_df_reindexed, groupby = 'fib_k=2_niche', method = 'wilcoxon', key_added = 'fib_k=2_niche_degs')

sc.pl.rank_genes_groups_dotplot(
    s2_r4_df_reindexed, groupby="fib_k=2_niche", standard_scale="var", n_genes=15, key = 'fib_k=2_niche_degs',
    save = 'fibs_leiden_tb_spatial_test.png')

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
plt.savefig('/Users/jacquelinechou/s2_r4_cxcl12_expression_all_sublining_fibroblasts.png')
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