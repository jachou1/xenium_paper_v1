import scanpy as sc
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import seaborn as sns

cohort_data = sc.read_h5ad('/Users/jacquelinechou/Downloads/240607_cohort_tcell_recluster_reindexed.h5ad')

sc.pl.umap(cohort_data, color = ['tissue'], save = 'cohort_tissue_distribution_umap.png')

# Let's plot a stacked barplot of the major cell-types per tissue
cell_types = cohort_data.obs['celltype_tier1_merged']
tissues = cohort_data.obs['tissue']

# Create a DataFrame with cell types and tissues
df = pd.DataFrame({'CellType': cell_types, 'Tissue': tissues})

# Create a count of cell types for each tissue
count_df = df.groupby(['Tissue', 'CellType']).size().unstack(fill_value=0)

# Save the dataframe as a csv file with the cell-type proportion per spatial group
count_df.to_csv('/Users/jacquelinechou/Downloads/cohort_level_major_celltypes_per_tissue.csv')
# Stacked barplot
# Plotting
ax = count_df.plot(kind='bar', stacked=True, figsize=(10, 7))
# Adding the legend explicitly
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Stacked Barplot of Cell Types in Each Tissue')
plt.xlabel('Tissue')
plt.ylabel('Number of Major Cell-types')
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin to make space for the legend
# plt.legend(loc='upper right', bbox_to_anchor=(1.5, 0.5))
plt.savefig('/Users/jacquelinechou/Downloads/cohort_level_absolute_cell_counts.png')
plt.show()

# That was for absolute cell counts. Let's do it for relative cells counts
relative_counts_celltypes = np.array(count_df) / np.array(count_df).sum(axis =1).reshape(-1, 1)
relative_counts_celltypes_df = pd.DataFrame(relative_counts_celltypes, columns = count_df.columns, index = count_df.index)

ax = relative_counts_celltypes_df.plot(kind='bar', stacked=True, figsize=(10, 7))
# Adding the legend explicitly
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Stacked Barplot of Cell Types in Each Tissue')
plt.xlabel('Tissue')
plt.ylabel('Proportion of Major Cell-types')
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the right margin to make space for the legend
# plt.legend(loc='upper right', bbox_to_anchor=(1.5, 0.5))
plt.savefig('/Users/jacquelinechou/Downloads/cohort_level_relative_cell_counts.png')
plt.show()

# I wonder if the lymphocyte compartment (B and T-cells) match up with the SLI score
immune_cell_proportion_per_tissue = (relative_counts_celltypes_df.loc[:, 'T-cells'] + relative_counts_celltypes_df.loc[:, 'Plasma cells']
                                     + relative_counts_celltypes_df.loc[:, 'Non-plasma B-cells'])

sli_3_tissue = ['slide1_r1', 'slide1_r3', 'slide1_r4', 'slide2_r3', 'slide2_r4']
immune_cell_proportion_per_tissue[immune_cell_proportion_per_tissue.index.isin(sli_3_tissue)]
sli_4_tissue = ['slide1_r5', 'slide2_r2', 'slide2_r5']

# What about correlation b/w lining fibroblasts compartment and lining hyperplasia? Although lining hyperplasia includes myeloid

# Let's look at the expression of ligands and receptors we want to focus on in all tissues
custom_genes = pd.read_csv('/Users/jacquelinechou/Documents/Xenium_custom_50_genes.csv', header = None)
custom_genes = np.array(custom_genes.iloc[:, 0])

# Let's plot the number of transcripts for each gene in aggregate across tissues
for tissue in cohort_data.obs['tissue'].unique():
    # Subset to tissue of interest
    tissue_adata = cohort_data[cohort_data.obs['tissue'] == tissue]
    # Make a multi-plot per tissue
    fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(5 * 5, 5 * 10))
    axes = axes.flatten()
    plot_idx = 0
    # Plot the number of transcripts detected for each custom gene as a histogram
    for gene in np.unique(custom_genes):
        ax = axes[plot_idx]
        # This plots the cells with zero transcripts (which makes it hard to appreciate 'signal')
        # gene_count = tissue_adata.layers['counts'][:, tissue_adata.var_names == gene].toarray().reshape(-1)
        gene_count = tissue_adata.layers['counts'][:, tissue_adata.var_names == gene].toarray().reshape(-1)
        nonzero_gene_count = gene_count[gene_count != 0]
        # print(gene_count.sum())
        ax.hist(nonzero_gene_count)
        # Create a swarm plot
        # sns.swarmplot(y=nonzero_gene_count, ax=ax, color='blue', alpha=0.7)
        ax.set_title(f'{tissue} - {gene}')
        ax.set_xlabel('Counts')
        ax.set_ylabel('Number of cells')
        plot_idx += 1
    plt.tight_layout()
    plt.savefig(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/{tissue}_custom_gene_nonzero_counts_aggregate_hist.png')

## Let's plot the expression of marker genes as log-transformed and normalized values in heatmap
# Then look at certain genes that we expect to be highly expressed in most cells of a specific type
endothelial_markers = ['ACKR1', 'VWF', 'CD34', 'PECAM1']
tcell_markers = ['CD3E', 'CD3D', 'TRAC']
nonplasma_bcell_markers = ['BANK1', 'BASP1', 'CD19', 'CD79A']
sublining_fib_markers = ['BASP1', 'THY1']
lining_fib_markers = ['PRG4','MET']
plasma_bcell_markers = ['CD79A', 'MZB1']
myeloid_markers = ['CD14', 'CD68']

celltypes = ['Endothelial cells', 'T-cells', 'Non-plasma B-cells', 'Sublining fibroblasts', 'Lining fibroblasts', 'Plasma cells', 'Myeloid cells']

# Let's look at the distribution of raw counts of each marker gene across cell-types in each tissue
list_of_markers = [endothelial_markers, tcell_markers, nonplasma_bcell_markers, sublining_fib_markers, lining_fib_markers,
                   plasma_bcell_markers, myeloid_markers]
all_markers = [item for sublist in list_of_markers for item in sublist]

# for i, major_celltype_genes in enumerate(list_of_markers):
# Let's include cells that do not express these markers per cell-type
# Let's subset the adata object to the genes of interest
subset_ad_celltype_markers = cohort_data[:, all_markers]
# Let's further subset to celltype of interest; MAY WANT TO REVISIT TO SEE EXPRESSION OF THESE GENES IN ALL CELL-TYPES
# subset_ad_celltype_markers = subset_ad_celltype_markers[subset_ad_celltype_markers.obs['celltype_tier1_merged'] == celltypes[i]]
log_counts_matrix = subset_ad_celltype_markers.X
log_counts_df = pd.DataFrame(log_counts_matrix.toarray(), columns = all_markers, index = subset_ad_celltype_markers.obs_names)
# print(counts_df.head())
log_combined_df = pd.concat([log_counts_df, subset_ad_celltype_markers.obs['tissue'],
                             subset_ad_celltype_markers.obs['celltype_tier1_merged']], axis = 1)
print(log_combined_df.columns)
# print(combined_df.head())
# Then create a long-form dataframe that has the per-cell gene counts and tissue labels
log_longform_df =pd.melt(log_combined_df, id_vars= ['tissue', 'celltype_tier1_merged'], value_vars = all_markers, var_name='gene',
                         value_name='expression')

log_longform_df.to_csv('/Users/jacquelinechou/Downloads/cohort_level_log_counts_per_major_celltype_marker_genes.csv')
# log_counts_per_cell_df.to_csv('/Users/jacquelinechou/Downloads/slide2_r4_raw_counts_per_cell_per_celltype.csv')

# Let's look at 1 tissue; take the mean expression of marker genes
log_marker_values = log_longform_df[log_longform_df.loc[:, 'tissue'] == 'slide2_r4'].groupby(['celltype_tier1_merged', 'gene'])['expression'].mean()

# Converts from a Series to a longform df
log_marker_values_df = log_marker_values.reset_index()
shortform_df = log_marker_values_df.pivot(index='celltype_tier1_merged', columns='gene', values='expression').reset_index()

# Change the order of the genes back to how I had them originally
# Reorder the columns
shortform_df = shortform_df[all_markers]
# Reset the index to make it a proper DataFrame
shortform_df.set_index('celltype_tier1_merged', inplace = True)

# Plot matrix
plt.figure(figsize=(8*2, 6))
sns.heatmap(shortform_df.apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True)
plt.title('Log-transformed and Normalized Counts \n (Slide2 Region4)')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.savefig('/Users/jacquelinechou/Downloads/heatmap_slide2_region4_log_transformed_and_normalized_counts_heatmap.png')

# Let's normalize across celltypes
normalized_shortform_df = shortform_df.div(shortform_df.max(axis=1), axis=0)
plt.figure(figsize=(8*2, 6))
sns.heatmap(normalized_shortform_df.apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True)
plt.title('Normalized Log(Median-transformed) Counts \n (Slide2 Region4)')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.savefig('/Users/jacquelinechou/Downloads/normalized_heatmap_slide2_region4_log_transformed_and_normalized_counts_heatmap.png')

## Plot normalized log-transformed and median-count-normalized data for each tissue
for tissue in log_longform_df.loc[:, 'tissue'].unique():
    log_marker_values = log_longform_df[log_longform_df.loc[:, 'tissue'] == tissue].groupby(['celltype_tier1_merged', 'gene'])['expression'].mean()
    # Converts from a Series to a longform df
    log_marker_values_df = log_marker_values.reset_index()
    # print(log_marker_values_df.head())
    shortform_df = log_marker_values_df.pivot(index='celltype_tier1_merged', columns='gene', values='expression').reset_index()
    # Reset the index to make it a proper DataFrame
    shortform_df.set_index('celltype_tier1_merged', inplace = True)
    # Change the order of the genes back to how I had them originally
    # Reorder the columns
    shortform_df = shortform_df[list(OrderedDict.fromkeys(all_markers))]
    # print(shortform_df.columns)
    normalized_shortform_df = shortform_df.div(shortform_df.max(axis=0), axis=1)
    plt.figure(figsize=(8 * 2, 6))
    sns.heatmap(normalized_shortform_df.apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True, fmt='.1f')
    plt.title(f'Normalized Log(Median-transformed) Counts \n ({tissue})')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.savefig(
        '/Users/jacquelinechou/Downloads/normalized_heatmap_' + f'{tissue}_log_transformed_and_normalized_counts_heatmap.png')

# Let's plot the median expression
for tissue in log_longform_df.loc[:, 'tissue'].unique():
    log_marker_values = log_longform_df[log_longform_df.loc[:, 'tissue'] == tissue].groupby(['celltype_tier1_merged', 'gene'])['expression'].median()
    # Converts from a Series to a longform df
    log_marker_values_df = log_marker_values.reset_index()
    # print(log_marker_values_df.head())
    shortform_df = log_marker_values_df.pivot(index='celltype_tier1_merged', columns='gene', values='expression').reset_index()
    # Reset the index to make it a proper DataFrame
    shortform_df.set_index('celltype_tier1_merged', inplace = True)
    # Change the order of the genes back to how I had them originally
    # Reorder the columns
    shortform_df = shortform_df[list(OrderedDict.fromkeys(all_markers))]
    # print(shortform_df.columns)
    normalized_shortform_df = shortform_df.div(shortform_df.max(axis=0), axis=1)
    plt.figure(figsize=(8 * 2, 6))
    sns.heatmap(normalized_shortform_df.apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True, fmt='.1f')
    plt.title(f'Normalized Log(Median-transformed) Counts \n ({tissue}): Median values')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.savefig(
        '/Users/jacquelinechou/Downloads/normalized_heatmap_' + f'{tissue}_median_log_transformed_and_normalized_counts_heatmap.png')


# EGFR expression in all tissues
gene_interest = 'EGFR'
for tissue in cohort_data.obs['tissue'].unique():
    # Subset to tissue of interest
    tissue_adata = cohort_data[cohort_data.obs['tissue'] == tissue]
    # Make a multi-plot per tissue
    plt.figure(figsize=(5, 5))
    # Plot the number of transcripts detected for each custom gene as a histogram
    # This plots the cells with zero transcripts (which makes it hard to appreciate 'signal')
    # gene_count = tissue_adata.layers['counts'][:, tissue_adata.var_names == gene].toarray().reshape(-1)
    gene_count = tissue_adata.layers['counts'][:, tissue_adata.var_names == gene_interest].toarray().reshape(-1)
    nonzero_gene_count = gene_count[gene_count != 0]
    # print(gene_count.sum())
    plt.hist(nonzero_gene_count)
    # Create a swarm plot
    # sns.swarmplot(y=nonzero_gene_count, ax=ax, color='blue', alpha=0.7)
    plt.title(f'{tissue} - {gene_interest}')
    plt.xlabel('Counts')
    plt.ylabel('Number of cells')
    plt.tight_layout()
    plt.savefig(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/{tissue}_EGFR_nonzero_counts_aggregate_hist.png')

# Let's plot the number of transcripts for each gene per cell-type across tissues
gene_interest = 'EGFR'
major_celltypes = cohort_data.obs['celltype_tier1_merged'].unique()
for tissue in cohort_data.obs['tissue'].unique():
    # Subset to tissue of interest
    tissue_adata = cohort_data[cohort_data.obs['tissue'] == tissue]
    # Make a multi-plot per tissue
    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize=(5 * 4, 5 * 2))
    axes = axes.flatten()
    plot_idx = 0
    for cell_type in major_celltypes:
        # print(cell_type)
        ax = axes[plot_idx]
        celltype_adata = tissue_adata[tissue_adata.obs['celltype_tier1_merged'] == cell_type]
        # Plot the number of transcripts detected for each custom gene as a histogram
        # This plots the cells with zero transcripts (which makes it hard to appreciate 'signal')
        # gene_count = tissue_adata.layers['counts'][:, tissue_adata.var_names == gene].toarray().reshape(-1)
        gene_count = celltype_adata.layers['counts'][:, celltype_adata.var_names == gene_interest].toarray().reshape(-1)
        nonzero_gene_count = gene_count[gene_count != 0]
        # print(gene_count.sum())
        ax.hist(nonzero_gene_count)
        # Create a swarm plot
        # sns.swarmplot(y=nonzero_gene_count, ax=ax, color='blue', alpha=0.7)
        ax.set_title(f'{tissue} - {gene_interest} in {cell_type}')
        ax.set_xlabel('Counts')
        ax.set_ylabel('Number of cells')
        plot_idx += 1
    plt.tight_layout()
    plt.savefig(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/{tissue}_EGFR_nonzero_counts_aggregate_hist_celltype.png')

# Should overlay of individual celltypes with the aggregate (tissue-level) plots

# Let's plot the expression of ligands and receptors in all tissue spatially with heatmap
gene_of_interest = 'EGFR'
plotting_factor = 300
for tissue in cohort_data.obs['tissue'].unique():
    tissue_adata = cohort_data[cohort_data.obs['tissue'] == tissue]
    tissue_adata_y_max = np.ceil(np.max(tissue_adata.obsm['spatial'][:, 1]))
    tissue_adata_y_min = np.floor(np.min(tissue_adata.obsm['spatial'][:, 1]))
    tissue_adata_x_max = np.ceil(np.max(tissue_adata.obsm['spatial'][:, 0]))
    tissue_adata_x_min = np.floor(np.min(tissue_adata.obsm['spatial'][:, 1]))
    y_range = tissue_adata_y_max - tissue_adata_y_min
    x_range = tissue_adata_x_max - tissue_adata_x_min
    plt.figure(figsize=(x_range / plotting_factor, y_range / plotting_factor), dpi=300)
    # show cells in tissue as grey centroids
    plt.scatter(x=tissue_adata.obsm['spatial'][:, 0], y=tissue_adata.obsm['spatial'][:, 1], c='grey', s=1)
    # overlay with dots for transcript count
    plt.scatter(x=np.array(tissue_adata.obsm['spatial'][:, 0]),
                y=np.array(tissue_adata.obsm['spatial'][:, 1]),
                s=1, cmap= 'inferno_r',
                c=(tissue_adata.layers['counts'][:, tissue_adata.var_names == gene_of_interest]).toarray().reshape(-1))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title(f'{gene_of_interest} expression in {tissue}')
    plt.colorbar(shrink=0.5)
    plt.savefig(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/{tissue}_{gene_of_interest}_expression_all.png', bbox_inches='tight')
    # plt.show()

# Let's plot the expression of genes as dots and cell segments as polygons for visualization, but it's just to see how the dots look
# in relation to the segments, so only informative in terms of QC for segmentation

# Comparison across tissues
gene_interest = 'CXCL12'
for tissue in cohort_data.obs['tissue'].unique():
    # Subset to tissue of interest
    tissue_adata = cohort_data[cohort_data.obs['tissue'] == tissue]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    ax = axes.flatten()
    plot_idx = 0
    for celltype in tissue_adata.obs['celltype_tier1_merged'].unique():
        if isinstance(celltype, str):  # Check if celltype is not nan
            print(f'Tissue: {tissue}, Cell Type: {celltype}')
            # Subset to cell type of interest
            tissue_celltype_ad = tissue_adata[tissue_adata.obs['celltype_tier1_merged'] == celltype]

            # Extract gene count data for the gene of interest
            gene_count = tissue_celltype_ad.layers['counts'][:,
                         tissue_celltype_ad.var_names == gene_interest].toarray().reshape(-1)

            # Filter out zero counts
            nonzero_gene_count = gene_count[gene_count != 0]

            # Print some debug information
            print(f'Gene count shape: {gene_count.shape}')
            print(f'Maximum non-zero count: {np.max(nonzero_gene_count)}')

            # Plot histogram of non-zero counts
            bins = np.arange(0, nonzero_gene_count.max() + 2) - 0.5
            ax[plot_idx].hist(nonzero_gene_count, bins=bins)
            ax[plot_idx].set_title(f'{tissue} - {gene_interest} in {celltype}')
            print(f'histogram_{tissue}_{celltype} done')
            plot_idx += 1

    plt.xlabel('Counts')
    plt.ylabel('Number of cells')
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        f'/Users/jacquelinechou/Downloads/cohort_level_analysis/{tissue}_CXCL12_nonzero_counts_by_celltype_hist.png')
    plt.close()

# Let's plot the expression of ligands and receptors in all tissue spatially with heatmap
gene_of_interest = 'CXCL12'
plotting_factor = 300
for tissue in cohort_data.obs['tissue'].unique():
    tissue_adata = cohort_data[cohort_data.obs['tissue'] == tissue]
    tissue_adata_y_max = np.ceil(np.max(tissue_adata.obsm['spatial'][:, 1]))
    tissue_adata_y_min = np.floor(np.min(tissue_adata.obsm['spatial'][:, 1]))
    tissue_adata_x_max = np.ceil(np.max(tissue_adata.obsm['spatial'][:, 0]))
    tissue_adata_x_min = np.floor(np.min(tissue_adata.obsm['spatial'][:, 1]))
    y_range = tissue_adata_y_max - tissue_adata_y_min
    x_range = tissue_adata_x_max - tissue_adata_x_min
    plt.figure(figsize=(x_range / plotting_factor, y_range / plotting_factor), dpi=300)
    # show cells in tissue as grey centroids
    plt.scatter(x=tissue_adata.obsm['spatial'][:, 0], y=tissue_adata.obsm['spatial'][:, 1], c='grey', s=1)
    # overlay with dots for transcript count
    plt.scatter(x=np.array(tissue_adata.obsm['spatial'][:, 0]),
                y=np.array(tissue_adata.obsm['spatial'][:, 1]),
                s=1, cmap= 'inferno_r',
                c=(tissue_adata.layers['counts'][:, tissue_adata.var_names == gene_of_interest]).toarray().reshape(-1))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title(f'{gene_of_interest} expression in {tissue}')
    plt.colorbar(shrink=0.5)
    plt.savefig(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/{tissue}_{gene_of_interest}_expression_all.png', bbox_inches='tight')
    # plt.show()
