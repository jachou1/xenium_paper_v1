import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.neighbors import KDTree

concat_ad = sc.read_h5ad('/Users/jacquelinechou/Downloads/240607_cohort_tcell_recluster_reindexed.h5ad')

# Let's just look at slide2_region5 (T/B aggregate heavy, good to test out spatial niche test)
slide2_region5_tissue = concat_ad[concat_ad.obs['tissue'] == 'slide2_r5']

# Create an adjacency matrix for all 141K cells with a radius = 100 um
def generate_neighbor_graph(single_cell_matrix, distance_threshold=100):
    ## Generates the GRAPH for all cells in data within 100 microns of each other
    kdt = KDTree(single_cell_matrix.obsm['spatial'], metric='euclidean')

    # get the nearest 100 meighbors using kdTree
    distances_matrix, neighbor_indices = kdt.query(single_cell_matrix.obsm['spatial'], k=151, return_distance=True)
    distances_matrix, neighbor_indices = distances_matrix[:, 1:], neighbor_indices[:, 1:]  # this removes the first column which is itself
    distances_mask = distances_matrix < distance_threshold
    return distances_matrix, neighbor_indices, distances_mask

distances_matrix, neighbor_graph, distances_mask = generate_neighbor_graph(slide2_region5_tissue)

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
positions = {i: tuple(coord) for i, coord in enumerate(slide2_region5_tissue.obsm['spatial'])}
for node, pos in positions.items():
    G.add_node(node, pos=pos)

# Add edges based on distances_mask and neighbor_indices
edge_indices = np.where(distances_mask == 1)
edges = [(i, neighbor_graph[i, k]) for i, k in zip(*edge_indices)]
G.add_edges_from(edges)

# Plot the neighbor connectivity: can't really discern the T/B aggregates, too many other cells nearby
plt.figure(figsize=(30*3, 30*3), dpi = 200)
nx.draw(G, positions, with_labels=False, node_color='lightblue', node_size=1, font_size=10, font_weight='bold', edge_color='gray')
plt.title("Spatial Network: Slide1 Region1")
plt.gca().set_aspect('equal')
plt.gca().invert_yaxis()
plt.savefig('/Users/jacquelinechou/Downloads/s2_r5/240619_s2_r5_spatial_network_90x90_200dpi.png');

# Get the labels of the neighbors and calculate the vector of cell-types per cell
# Run k-means clustering on those vectors of cell-types to get structures

# Let's get the cell-type labels per cell using neighbor graph
subset_list = [slide2_region5_tissue.obs['celltype_tier1_merged'].iloc[neighbors].values for neighbors in filtered_neighbors]

celltype_labels = slide2_region5_tissue.obs['celltype_tier1_merged'].unique()

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
# slide2_region5_tissue.obs['spatial_neighborhood_k_2'] = labels

# Let's plot the spatial groups identified via k-means clustering
# plt.figure(figsize=(20, 30), dpi= 300)
# for color, group in slide2_region5_tissue.obs.groupby(['spatial_neighborhood_k_2']):
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
np.save('/Users/jacquelinechou/Downloads/s2_r5/s2_r5_centroids5_kmeans5.npy', centroids5)

# Add these spatial neighborhood_k=2 to the anndata object
slide2_region5_tissue.obs['spatial_neighborhood_k_5'] = labels5

from matplotlib import colors as mcolors
slide2_region5_tissue.obs['spatial_x'] = slide2_region5_tissue.obsm['spatial'][:, 0]
slide2_region5_tissue.obs['spatial_y'] = slide2_region5_tissue.obsm['spatial'][:, 1]

plt.figure(figsize=(20, 30), dpi= 300)
for color, group in slide2_region5_tissue.obs.groupby(['spatial_neighborhood_k_5']):
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
plt.savefig('/Users/jacquelinechou/Downloads/s2_r5/s2_r5_test_spatial_plot_k_5.png')

# Let's make the stacked barplot of the spatial clusters identified using kmeans
spatial_neighborhoods_df = pd.DataFrame(centroids5, columns = celltype_labels)

spatial_neighborhoods_df['Spatial_neighborhood'] = ['Spatial cluster 0', 'Spatial cluster 1', 'Spatial cluster 2',
                                                    'Spatial cluster 3', 'Spatial cluster 4']
# Save the dataframe as a csv file with the cell-type proportion per spatial group
spatial_neighborhoods_df.to_csv('/Users/jacquelinechou/Downloads/s2_r5/slide2_region5_kmeans_5_df.csv')

# Stacked barplot
spatial_neighborhoods_df.plot(x = 'Spatial_neighborhood', kind = 'bar', stacked=True)
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 0.5))
plt.savefig('/Users/jacquelinechou/Downloads/s2_r5/240619_s2_r5_celltype_composition_barplot_kmeans_5.png', bbox_inches='tight')

# Let's create a new column in the anndata wherein we create a boolean mask for whether sublining fibroblasts are within spatial neighborhood, k=2
# slide2_region5_tissue.obs['fib_spatial_group=1_niche'] = np.where((slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Sublining fibroblasts') &
#                                                    (slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == 1), '1', '0')

# Let's run differential gene expression analysis using scanpy
# sc.tl.rank_genes_groups(slide2_region5_tissue, groupby = 'fib_spatial_group=1_niche', method = 'wilcoxon', key_added = 'fib_sp=1_niche_degs')
#
# sc.pl.rank_genes_groups_dotplot(
#     slide2_region5_tissue, groupby="fib_spatial_group=1_niche", standard_scale="var", n_genes=15, key = 'fib_sp=1_niche_degs',
#     save = 's1_r3_fibs_leiden_tb_spatial_group_degs_dotplot.png')

# So it looks like the DEGs for sublining fibroblasts inside T/B aggregates is also
# expressed in sublining fiborblasts elsewhere, but not as highly expressed?

# Let's look at CXCL12 expression in this tissue
# plt.figure(figsize = (40, 20), dpi = 200)
# plt.scatter(x = s2_r4_df_reindexed.obsm['spatial'][:, 0], y = s2_r4_df_reindexed.obsm['spatial'][:, 1], s =1 ,
#             c = s2_r4_df_reindexed.X[:, s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'CXCL12'].toarray());
# plt.gca().invert_yaxis()
# plt.gca().set_aspect('equal')
# plt.colorbar()
# # plt.savefig('/Users/jacquelinechou/s2_r3_cxcl12_expression_all_cells.png')
# plt.show()
#
# # Let's plot CXCL12 expression in sublining fibroblasts only
# plt.figure(figsize = (40*1.5, 20*1.5), dpi = 300)
# plt.scatter(x = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 0]),
#                 y = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 1]), s =1 ,
#             c = np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
#             s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'CXCL12']))
# plt.gca().invert_yaxis()
# plt.gca().set_aspect('equal')
# plt.title('CXCL12 expression in sublining fibroblasts only (S2_R4)')
# plt.colorbar(shrink = 0.25)
# plt.savefig('/Users/jacquelinechou/s2_r3_cxcl12_expression_all_sublining_fibroblasts.png')
# plt.show()

# Let's plot the spatial_neighborhoods and the CXCL12 expression in sublining fibroblasts overlayed
# plt.figure(figsize = (40/2, 20/2), dpi = 300)
# plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
#             y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
#             c = 'lightskyblue', s = 1)
# plt.scatter(x = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 0]),
#                 y = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 1]), s =1 , cmap =
#                                                                                                                                        'inferno_r',
#             c = np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
#             s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'CXCL12']))
# plt.gca().invert_yaxis()
# plt.gca().set_aspect('equal')
# plt.title('CXCL12 expression in sublining fibroblasts only  \n spatial_neighborhood = 2 (S2_R4)')
# plt.colorbar(shrink = 0.5)
# plt.savefig('/Users/jacquelinechou/Downloads/test_s2_r3_cxcl12_expression_all_sublining_fibroblasts.png')
# plt.show()
#
# plt.figure(figsize = (40/2, 20/2), dpi = 300)
# plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
#             y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
#             c = 'lightskyblue', s = 1)
# plt.scatter(x = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 0]),
#                 y = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 1]), s =1 , cmap = 'inferno_r',
#             c = np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
#             s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'CCL2']))
# plt.gca().invert_yaxis()
# plt.gca().set_aspect('equal')
# plt.title('CCL2 expression in sublining fibroblasts only  \n spatial_neighborhood = 2 (S2_R4)')
# plt.colorbar(shrink = 0.5)
# plt.savefig('/Users/jacquelinechou/Downloads/test_s2_r4_ccl2_expression_all_sublining_fibroblasts.png')
# plt.show()

# plt.figure(figsize = (40/2, 20/2), dpi = 300)
# plt.scatter(x=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 0],
#             y=s2_r4_df_reindexed[s2_r4_df_reindexed.obs['spatial_neighborhood_k_2'] == 1].obsm['spatial'][:, 1],
#             c = 'lightskyblue', s = 1)
# plt.scatter(x = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 0]),
#                 y = np.array(s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].obsm['spatial'][:, 1]), s =1 , cmap =
#                                                                                                                                        'inferno_r',
#             c = np.array(s2_r4_df_reindexed.layers['counts'][(s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'),
#             s2_r4_df_reindexed[s2_r4_df_reindexed.obs['leiden_res_0_3'] == '1'].var_names == 'PTGDS']))
# plt.gca().invert_yaxis()
# plt.gca().set_aspect('equal')
# plt.title('PTGDS expression in sublining fibroblasts only  \n spatial_neighborhood = 2 (S2_R4)')
# plt.colorbar(shrink = 0.5)
# plt.savefig('/Users/jacquelinechou/Downloads/test_s2_r4_ptgds_expression_all_sublining_fibroblasts.png')
# plt.show()

slide2_region5_tissue.write('/Users/jacquelinechou/Downloads/s2_r5/240619_s2_r5_post_kmeans_tissue.h5ad')

# Uncomment to make plots; but need to look at results of kmeans clustering to determine which spatial group to look at
top_fibs_degs_spatial_k2 = ['CXCL12', 'VCAN', 'FBN1', 'CXCL13', 'PTGDS', 'TNC', 'PDGFRA', 'FBLN1', 'GLI3']
for gene in np.arange(len(top_fibs_degs_spatial_k2)):
    gene_of_interest = top_fibs_degs_spatial_k2[gene]
    plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
    plt.scatter(x=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == tb_spatial_group].obsm['spatial'][:, 0],
                y=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == tb_spatial_group].obsm['spatial'][:, 1],
                c='lightskyblue', s=1)
    plt.scatter(x=np.array(slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Sublining fibroblasts'].obsm['spatial'][:, 0]),
                y=np.array(slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Sublining fibroblasts'].obsm['spatial'][:, 1]),
                s=1, cmap= 'inferno_r',
                c=np.array(slide2_region5_tissue.layers['counts'][(slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Sublining fibroblasts'),
                slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Sublining fibroblasts'].var_names == gene_of_interest]))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title(f'{gene_of_interest} expression in sublining fibroblasts only  \n spatial_neighborhood = 4 (S1_R5)')
    plt.colorbar(shrink=0.5)
    plt.savefig(f'/Users/jacquelinechou/Downloads/s1_r5/test_s1_r5_{gene_of_interest}_expression_all_sublining_fibroblasts.png')
    plt.show()

# Let's plot all top 15 genes (add all raw counts)
top_fibs_15_degs_spatial_k2 = ['CXCL12', 'PTGDS', 'VCAN', 'TNC', 'FBN1', 'CTSK', 'FBLN1', 'GLI3', 'CTSK',
                               'COL5A2', 'CRISPLD2', 'PCOLCE', 'CCR7', 'PDGFRB']
leiden_condition = slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Sublining fibroblasts'
genes_condition = slide2_region5_tissue.var_names.isin(top_fibs_15_degs_spatial_k2)
top15_gene_count = slide2_region5_tissue.layers['counts'][leiden_condition][:, genes_condition].sum(axis = 1)
# Extract the spatial data for the specific condition
spatial_data = slide2_region5_tissue[leiden_condition].obsm['spatial']

plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
# Plot the spatial neighborhood, k = 2
plt.scatter(x=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == tb_spatial_group].obsm['spatial'][:, 0],
            y=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == tb_spatial_group].obsm['spatial'][:, 1],
            c='lightskyblue', s=1)
plt.scatter(x=np.array(slide2_region5_tissue[leiden_condition].obsm['spatial'][:, 0]),
            y=np.array(slide2_region5_tissue[leiden_condition].obsm['spatial'][:, 1]),
            s=1, cmap= 'inferno_r', c= np.array(top15_gene_count).reshape(-1))
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('Top 15 DEGs expression in sublining fibroblasts only  \n spatial_neighborhood = 4 (S1_R5)')
plt.colorbar(shrink=0.5)
plt.savefig(f'/Users/jacquelinechou/Downloads/s1_r5/test_s1_r5_top_15_gene_expression_all_sublining_fibroblasts.png')
plt.show()

# # Let's create a new column in the anndata wherein we create a boolean mask for whether sublining fibroblasts are within spatial neighborhood, k=2
# slide2_region5_tissue.obs['tcell_spatial_group=2_niche'] = np.where((slide2_region5_tissue.obs['celltype_tier1_merged'] == 'T-cells') &
#                                                    (slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == 2), '1', '0')
#
# # Let's run differential gene expression analysis using scanpy
# sc.tl.rank_genes_groups(slide2_region5_tissue, groupby = 'tcell_spatial_group=2_niche', method = 'wilcoxon', key_added = 'tcell_sp=2_niche_degs')
#
# sc.pl.rank_genes_groups_dotplot(
#     slide2_region5_tissue, groupby="tcell_spatial_group=2_niche", standard_scale="var", n_genes=25, key = 'tcell_sp=2_niche_degs',
#     save = '240612_s2_r4_tcell_leiden_tb_spatial_test_degs_dotplot.png')
#
# # How are the non-plasma B-cells in these aggregates different from other non-plasma B-cells?
# slide2_region5_tissue.obs['nonplasma_bcell_spatial_group=2_niche'] = np.where((slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Non-plasma B-cells') &
#                                                    (slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == 2), '1', '0')
# # Let's run differential gene expression analysis using scanpy
# sc.tl.rank_genes_groups(slide2_region5_tissue, groupby = 'nonplasma_bcell_spatial_group=2_niche', method = 'wilcoxon', key_added = 'nonplasma_bcell_sp=2_niche_degs')
#
# sc.pl.rank_genes_groups_dotplot(
#     slide2_region5_tissue, groupby="nonplasma_bcell_spatial_group=2_niche", standard_scale="var", n_genes=15, key = 'nonplasma_bcell_sp=2_niche_degs',
#     save = '240612_s2_r4_nonplasma_bcell_leiden_tb_spatial_test_degs_dotplot.png')
#
# # Instead of just visualizing a few DEGs, we can get a dataframe with the DEGs
# df = sc.get.rank_genes_groups_df(slide2_region5_tissue, group='1', key='tcell_sp=2_niche_degs')
#
# # with plt.rc_context({"figure.figsize": (4.5, 3)}):
# #     sc.pl.violin(s2_r4_df_reindexed, ["TRAC", "MS4A1", "PTPRC", "CXCL13"], groupby="tcell_k=2_niche")
#
# # Let's look at the spatial localization of these DEGs wrt their cell-type location
# top_tcells_degs_spatial_k2 = list(df['names'][0:10])
# for gene in np.arange(len(top_tcells_degs_spatial_k2)):
#     gene_of_interest = top_tcells_degs_spatial_k2[gene]
#     print(gene_of_interest)
#     plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
#     plt.scatter(x=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == 2].obsm['spatial'][:, 0],
#                 y=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == 2].obsm['spatial'][:, 1],
#                 c='lightskyblue', s=1)
#     plt.scatter(x=np.array(slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'T-cells'].obsm['spatial'][:, 0]),
#                 y=np.array(slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'T-cells'].obsm['spatial'][:, 1]),
#                 s=1, cmap= 'inferno_r', alpha = 0.5,
#                 c=np.array(slide2_region5_tissue.layers['counts'][(slide2_region5_tissue.obs['celltype_tier1_merged'] == 'T-cells'),
#                 slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'T-cells'].var_names == gene_of_interest]))
#     plt.gca().invert_yaxis()
#     plt.gca().set_aspect('equal')
#     plt.title(f'{gene_of_interest} expression in T-cells only  \n spatial_neighborhood = 2 (S2_R4)')
#     plt.colorbar(shrink=0.5)
#     plt.savefig(f'/Users/jacquelinechou/Downloads/240618_s2_r3_{gene_of_interest}_expression_all_tcells.png')
#
# df_bcells = sc.get.rank_genes_groups_df(slide2_region5_tissue, group='1', key='nonplasma_bcell_sp=2_niche_degs')
#
# top_nonplasma_cells_degs_spatial_k2 = list(df_bcells['names'][0:15])
# for gene in np.arange(len(top_nonplasma_cells_degs_spatial_k2)):
#     gene_of_interest = top_nonplasma_cells_degs_spatial_k2[gene]
#     print(gene_of_interest)
#     plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
#     plt.scatter(x=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == 2].obsm['spatial'][:, 0],
#                 y=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == 2].obsm['spatial'][:, 1],
#                 c='lightskyblue', s=1)
#     plt.scatter(x=np.array(slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Non-plasma B-cells'].obsm['spatial'][:, 0]),
#                 y=np.array(slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Non-plasma B-cells'].obsm['spatial'][:, 1]),
#                 s=1, cmap= 'inferno_r', alpha = 0.5,
#                 c=np.array(slide2_region5_tissue.layers['counts'][(slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Non-plasma B-cells'),
#                 slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Non-plasma B-cells'].var_names == gene_of_interest]))
#     plt.gca().invert_yaxis()
#     plt.gca().set_aspect('equal')
#     plt.title(f'{gene_of_interest} expression in Non-plasma B-cells only  \n spatial_neighborhood = 2 (S1_R4)')
#     plt.colorbar(shrink=0.5)
#     plt.savefig(f'/Users/jacquelinechou/Downloads/240618_s2_r3_{gene_of_interest}_expression_all_non_plasma_bcells.png')
#
#
# Let's run through a for loop for all 3 cell-types (Sublining fibroblasts, T-cells, and non-plasma B-cells)
celltypes_interest = ['Sublining fibroblasts', 'T-cells', 'Non-plasma B-cells']
tb_spatial_group = int(0)
tissue = 'slide2_region5'
for celltype in celltypes_interest:
    slide2_region5_tissue.obs[f'{celltype}_spatial_group={tb_spatial_group}_niche'] = np.where(
        (slide2_region5_tissue.obs['celltype_tier1_merged'] == celltype) &
        (slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == tb_spatial_group), '1', '0')

    # Let's run differential gene expression analysis using scanpy
    sc.tl.rank_genes_groups(slide2_region5_tissue, groupby=f'{celltype}_spatial_group={tb_spatial_group}_niche', method='wilcoxon',
                            key_added=f'{celltype}_sp={tb_spatial_group}_niche_degs')

    sc.pl.rank_genes_groups_dotplot(
        slide2_region5_tissue, groupby=f'{celltype}_spatial_group={tb_spatial_group}_niche', standard_scale="var", n_genes=15,
        key = f'{celltype}_sp={tb_spatial_group}_niche_degs',
        save=f'{tissue}_{celltype}_leiden_tb_spatial_group_degs_dotplot.png')

# Save h5ad file so far
slide2_region5_tissue.write('/Users/jacquelinechou/Downloads/s2_r6/240619_s2_r6_post_kmeans_tissue.h5ad')

# df_bcells = sc.get.rank_genes_groups_df(slide2_region5_tissue, group='1', key='Non-plasma B-cells_sp=1_niche_degs')
#
# top_nonplasma_cells_degs_spatial_k2 = list(df_bcells['names'][0:15])
# for gene in np.arange(len(top_nonplasma_cells_degs_spatial_k2)):
#     gene_of_interest = top_nonplasma_cells_degs_spatial_k2[gene]
#     print(gene_of_interest)
#     plt.figure(figsize=(40 / 2, 20 / 2), dpi=300)
#     plt.scatter(x=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == tb_spatial_group].obsm['spatial'][:, 0],
#                 y=slide2_region5_tissue[slide2_region5_tissue.obs['spatial_neighborhood_k_5'] == tb_spatial_group].obsm['spatial'][:, 1],
#                 c='lightskyblue', s=1)
#     plt.scatter(x=np.array(slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Non-plasma B-cells'].obsm['spatial'][:, 0]),
#                 y=np.array(slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Non-plasma B-cells'].obsm['spatial'][:, 1]),
#                 s=1, cmap= 'inferno_r', alpha = 0.5,
#                 c=np.array(slide2_region5_tissue.layers['counts'][(slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Non-plasma B-cells'),
#                 slide2_region5_tissue[slide2_region5_tissue.obs['celltype_tier1_merged'] == 'Non-plasma B-cells'].var_names == gene_of_interest]))
#     plt.gca().invert_yaxis()
#     plt.gca().set_aspect('equal')
#     plt.title(f'{gene_of_interest} expression in Non-plasma B-cells only  \n spatial_neighborhood = 2 (S1_R3)')
#     plt.colorbar(shrink=0.5)
#     plt.savefig(f'/Users/jacquelinechou/Downloads/240618_s2_r3_{gene_of_interest}_expression_all_non_plasma_bcells.png')