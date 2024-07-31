# Let's look at the expression of ligands and receptors of interest on a tissue-level
# Let's look at the raw counts
# Then look at the z-scored values
# TODO: Do a similar analysis using AMP data (need to revisit what format that data is in)
# TODO: Comparison across tissues

import scanpy as sc
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from shapely.geometry import Polygon, Point
import seaborn as sns
from matplotlib.colors import ListedColormap

amp2_qc_mrna = pd.read_csv('/Users/jacquelinechou/Downloads/qc_mrna_amp_subset_xenium_genes_by_celltype.csv')
allcells_ref = pd.read_csv('/Users/jacquelinechou/Downloads/all_cells_metadata.csv')

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

# Let's include cells that do not express these markers per cell-type
# Let's subset the adata object to the genes of interest
subset_amp2_marker_genes = amp2_qc_mrna.loc[all_markers, :]

unique_celltypes = subset_amp2_marker_genes.columns.unique()
subset_amp2_marker_genes_grouped = subset_amp2_marker_genes.groupby(by= subset_amp2_marker_genes.columns, axis = 1).mean() # generates a gene x celltype df
normalized_amp_ref = subset_amp2_marker_genes_grouped.div(subset_amp2_marker_genes_grouped.max(axis = 1), axis = 0)

# Let's normalize across celltypes
plt.figure(figsize=(8*2, 6))
sns.heatmap(normalized_amp_ref.T.apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True, fmt = '.1f')
plt.title('Normalized Log(CPM) Counts (scRNA reference)')
plt.xlabel('Marker Genes')
plt.ylabel('Broad cell-types')
plt.savefig('/Users/jacquelinechou/Downloads/amp_log_transformed_and_normalized_counts_heatmap.png')

# Let's look at the specific cell-types I generated from analyzing Xenium data
# Let's add another column to the allcell_ref with these Xenium-derived celltype labels
mapping_to_xenium_dict = {
    'sublining': 'Sublining fibroblast', #Match 'sublining' first
    'E-': 'Endothelial cell',
    'B-0': 'Non-plasma B-cell',
    'B-1': 'Non-plasma B-cell',
    'B-3': 'Non-plasma B-cell',
    'B-4': 'Non-plasma B-cell',
    'B-5': 'Non-plasma B-cell',
    'B-8': 'Non-plasma B-cell',
    'lining': 'Lining fibroblast',
    'NK-': ('NK cell', 'start'),
    'plasma': 'Plasma cell',
    'M-': 'Myeloid cell',
    'T-': 'T-cell',
}

# If a column value doesn't contain the above strings, then it is a Non-plasma B-cell
default_position = 'any'
def map_value(text, mapping_to_dict, default_position):
    for old_value, new_value in mapping_to_dict.items():
        if isinstance(new_value, tuple):
            new_value, position = new_value
        else:
            position = default_position

            # Check the text based on the position
        if position == 'start' and text.startswith(old_value):
            return new_value
        # elif position == 'end' and text.endswith(old_value):
        #     return new_value
        elif position == 'any' and old_value in text:
            return new_value


# Dreate the new column using the apply function
allcells_ref['Xenium-derived celltypes'] = allcells_ref['cluster_name'].apply(lambda x: map_value(x, mapping_to_xenium_dict, default_position))

# Now let's use the new column to replace the column names in the amp2_qc_mrna_grouped
remapping_columns = allcells_ref[['cell', 'Xenium-derived celltypes']].drop_duplicates()
remapping_columns_dict = dict(zip(remapping_columns['cell'], remapping_columns['Xenium-derived celltypes']))

# Let's read in the matrix with the cell_id labels in the columns
amp2_qc_mrna_cell_ids = pd.read_csv('/Users/jacquelinechou/Downloads/qc_mrna_amp_subset_xenium_genes.csv')
amp2_qc_mrna_cell_ids.set_index('Unnamed: 0', inplace=True)
amp2_qc_mrna_cell_ids.rename(columns = remapping_columns_dict, inplace = True)

# Now let's remake the df
subset_amp2_qc_marker_genes = amp2_qc_mrna_cell_ids.loc[all_markers, :]

unique_celltypes = subset_amp2_qc_marker_genes.columns.unique()
subset_amp2_qc_marker_genes_grouped = subset_amp2_qc_marker_genes.groupby(by= subset_amp2_qc_marker_genes.columns, axis = 1).mean() # generates a gene x celltype df
normalized_amp_ref_2 = subset_amp2_qc_marker_genes_grouped.div(subset_amp2_qc_marker_genes_grouped.max(axis = 1), axis = 0)

normalized_amp_ref_2_dedup = normalized_amp_ref_2[~normalized_amp_ref_2.index.duplicated(keep='first')]

# Let's save the df with the Xenium celltype labels
amp2_qc_mrna_cell_ids.to_csv('/Users/jacquelinechou/Downloads/AMP_reference_analysis_for_Xenium_data/qc_mrna_gene_cellid_matrix.csv')

# Let's normalize across celltypes
plt.figure(figsize=(8*2, 6))
sns.heatmap(normalized_amp_ref_2_dedup.T.apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True, fmt = '.1f')
plt.title('Normalized Log(CPM) Counts (scRNA reference)')
plt.xlabel('Marker Genes')
plt.ylabel('Broad cell-types')
plt.savefig('/Users/jacquelinechou/Downloads/amp_log_transformed_and_normalized_counts_heatmap_with_xenium_celllabels.png')

# Let's look at a subset of the custom genes
custom_ligands = ['IL1B', 'IL6', 'RSPO2', 'RSPO3', 'WNT5A', 'LRP5', 'LRP6', 'HBEGF', 'HGF']
custom_receptors = ['IL6R', 'CCR4', 'CXCR5', 'CXCR6', 'ACKR3', 'EGFR']

genes_of_interest = np.concatenate([custom_ligands, custom_receptors])

# Let's pull out the genes of interest from matrix and plot what the normalized values look like
subset_amp2_qc_marker_genes = amp2_qc_mrna_cell_ids.loc[genes_of_interest, :]
subset_amp2_qc_marker_genes_grouped = subset_amp2_qc_marker_genes.groupby(by= subset_amp2_qc_marker_genes.columns, axis = 1).mean() # generates a gene x celltype df
normalized_amp_ref_2 = subset_amp2_qc_marker_genes_grouped.div(subset_amp2_qc_marker_genes_grouped.max(axis = 1), axis = 0)

# normalized_amp_ref_2_dedup = normalized_amp_ref_2[~normalized_amp_ref_2.index.duplicated(keep='first')]

# Let's normalize across celltypes
plt.figure(figsize=(8*2, 6))
sns.heatmap(normalized_amp_ref_2.T.apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True, fmt = '.1f')
plt.title('Normalized Log(CPM) Counts (scRNA reference)')
plt.xlabel('Select Ligand / Receptor Genes')
plt.ylabel('Broad cell-types')
plt.savefig('/Users/jacquelinechou/Downloads/amp_log_transformed_and_normalized_counts_heatmap_with_xenium_celllabels_ligand_receptors.png')

# Let's plot the relative expression of these ligands and receptors in the cohort data
cohort_data = sc.read_h5ad('/Users/jacquelinechou/Downloads/240607_cohort_tcell_recluster_reindexed.h5ad')
subset_ad_celltype_markers = cohort_data[:, genes_of_interest]
# Let's further subset to celltype of interest; MAY WANT TO REVISIT TO SEE EXPRESSION OF THESE GENES IN ALL CELL-TYPES
# subset_ad_celltype_markers = subset_ad_celltype_markers[subset_ad_celltype_markers.obs['celltype_tier1_merged'] == celltypes[i]]
log_counts_matrix = subset_ad_celltype_markers.X
log_counts_df = pd.DataFrame(log_counts_matrix.toarray(), columns = genes_of_interest, index = subset_ad_celltype_markers.obs_names)
# print(counts_df.head())
log_combined_df = pd.concat([log_counts_df, subset_ad_celltype_markers.obs['tissue'],
                             subset_ad_celltype_markers.obs['celltype_tier1_merged']], axis = 1)
print(log_combined_df.columns)
# print(combined_df.head())
# Then create a long-form dataframe that has the per-cell gene counts and tissue labels
log_longform_df =pd.melt(log_combined_df, id_vars= ['tissue', 'celltype_tier1_merged'], value_vars = genes_of_interest, var_name='gene',
                         value_name='expression')

log_longform_df.to_csv('/Users/jacquelinechou/Downloads/cohort_level_log_counts_per_major_celltype_ligand_receptor_genes.csv')
# log_counts_per_cell_df.to_csv('/Users/jacquelinechou/Downloads/slide2_r4_raw_counts_per_cell_per_celltype.csv')

# Let's look at 1 tissue; take the mean expression of marker genes
log_marker_values = log_longform_df[log_longform_df.loc[:, 'tissue'] == 'slide2_r4'].groupby(['celltype_tier1_merged', 'gene'])['expression'].mean()

# Converts from a Series to a longform df
log_marker_values_df = log_marker_values.reset_index()
shortform_df = log_marker_values_df.pivot(index='celltype_tier1_merged', columns='gene', values='expression').reset_index()

# Change the order of the genes back to how I had them originally
# Reorder the columns
# shortform_df = shortform_df[genes_of_interest]
# Reset the index to make it a proper DataFrame
shortform_df.set_index('celltype_tier1_merged', inplace = True)

# Let's normalize based on the largest log expression across celltypes per gene
normalized_shortform_df = shortform_df.div(shortform_df.max(axis = 0), axis = 1)

# Let's reorder the columns names so they correspond to the heatmap for scRNA
normalized_shortform_df = normalized_shortform_df[genes_of_interest]
# Plot matrix
plt.figure(figsize=(8*2, 6))
sns.heatmap(normalized_shortform_df.apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True)
plt.title('Log-transformed and Normalized Counts \n (Slide2 Region4)')
plt.xlabel('Select Ligand / Receptor Genes')
plt.ylabel('Broad cell-types')
plt.savefig('/Users/jacquelinechou/Downloads/heatmap_slide2_region4_log_transformed_and_normalized_counts_heatmap_ligands_receptors_relative_exp.png')