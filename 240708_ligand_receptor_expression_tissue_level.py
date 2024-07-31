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

cohort_data = sc.read_h5ad('/Users/jacquelinechou/Downloads/240607_cohort_tcell_recluster_reindexed.h5ad')
custom_ligands = ['IL1B', 'IL6', 'RSPO2', 'RSPO3', 'WNT5A', 'LRP5', 'LRP6', 'HBEGF', 'HGF']
custom_receptors = ['IL6R', 'CCR4', 'CXCR5', 'CXCR6', 'ACKR3', 'EGFR']

column_names = np.concatenate([['major_celltype'], custom_ligands, custom_receptors])
raw_counts_df = pd.DataFrame(columns=column_names)
# Set the index of df to the major celltype

# So let's fill in the major_celltype column
raw_counts_df['major_celltype'] = cohort_data.obs['celltype_tier1_merged'].unique()
raw_counts_df.set_index('major_celltype', inplace=True)

raw_counts_per_cell_df = pd.DataFrame(columns = raw_counts_df.columns, index= raw_counts_df.index.dropna())
# Let's look at slide2_r4
slide2_r4_ad = cohort_data[cohort_data.obs['tissue'] == 'slide2_r4']
# Now let's use the index to filter the adata and calculate the raw values of each ligand and receptor of interest
for major_celltype in raw_counts_df.index:
    # For each gene, add up the counts per celltype
    tissue_data_celltype = slide2_r4_ad[slide2_r4_ad.obs['celltype_tier1_merged'] == major_celltype]
    for gene in raw_counts_df.columns:
        total_gene_count = tissue_data_celltype.layers['counts'][:, tissue_data_celltype.var_names == gene].sum()
        raw_counts_df.loc[raw_counts_df.index == major_celltype, gene] = total_gene_count
        # Let's divide by the number of cells per cell-type to get average transcripts per cell
        number_of_cells_of_type = tissue_data_celltype.shape[0]
        # print(number_of_cells_of_type)
        if number_of_cells_of_type == 0:
            raw_counts_per_cell_df.loc[raw_counts_per_cell_df.index == major_celltype, gene] = np.nan
        else:
            raw_counts_per_cell_df.loc[raw_counts_per_cell_df.index == major_celltype, gene] = (total_gene_count / number_of_cells_of_type)
raw_counts_df.to_csv('/Users/jacquelinechou/Downloads/slide2_r4_raw_counts_per_major_celltype.csv')
raw_counts_per_cell_df.to_csv('/Users/jacquelinechou/Downloads/slide2_r4_raw_counts_per_cell_per_celltype.csv')

# Plot matrix
plt.figure(figsize=(8*2, 6))
sns.heatmap(raw_counts_df.dropna().apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True, fmt='d', )
plt.title('Raw Counts (Slide2 Region4')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.savefig('/Users/jacquelinechou/Downloads/matrix_plot_slide2_region4_raw_counts_matrixplot.png')

plt.figure(figsize=(8*2, 6))
sns.heatmap(raw_counts_per_cell_df.dropna().apply(pd.to_numeric, errors='coerce'), cmap='inferno_r', annot=True, fmt='.2f')
plt.title('Average Number of Transcript Counts Per Cell (Slide2 Region4) \n'
          'including zero-cells')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.savefig('/Users/jacquelinechou/Downloads/matrix_plot_slide2_region4_average_raw_counts_per_cell_heatmap_with_zeroes.png')

raw_counts_per_cell_df_nonzero = pd.DataFrame(columns = raw_counts_per_cell_df.columns, index = raw_counts_per_cell_df.index)

for major_celltype in raw_counts_df.index:
    # For each gene, add up the counts per celltype
    tissue_data_celltype = slide2_r4_ad[slide2_r4_ad.obs['celltype_tier1_merged'] == major_celltype]
    for gene in raw_counts_df.columns:
        total_gene_count = tissue_data_celltype.layers['counts'][:, tissue_data_celltype.var_names == gene].sum()
        # Let's divide by the number of non-zero cells per cell-type to get average transcripts per cell (REMOVING THE CELLS WITH ZERO TRANSCRIPTS)
        nonzero_cell_count = (tissue_data_celltype.layers['counts'][:, tissue_data_celltype.var_names == gene] != 0).sum()
        # print(number_of_cells_of_type)
        if nonzero_cell_count == 0:
            raw_counts_per_cell_df_nonzero.loc[raw_counts_per_cell_df.index == major_celltype, gene] = np.nan
        else:
            raw_counts_per_cell_df_nonzero.loc[raw_counts_per_cell_df.index == major_celltype, gene] = (total_gene_count / nonzero_cell_count)

# After you get the raw dataframe, save it, then create a z-scored dataframe (z-scored by mean transcripts across all cells)
# NOT including cells with zero counts
# z-score = mean(Gene_a in celltype_1) - mean(Gene_a in all cells) / standard_dev(Gene_a in all cells)
z_score_nonzero_df = pd.DataFrame(columns = raw_counts_per_cell_df_nonzero.columns, index = raw_counts_per_cell_df_nonzero.index)
average_count_per_cell = np.empty(len(z_score_nonzero_df.columns))
std_count_per_cell = np.empty(len(z_score_nonzero_df.columns))
for idx, gene in enumerate(z_score_nonzero_df.columns):
    total_gene_count = tissue_data_celltype.layers['counts'][:, tissue_data_celltype.var_names == gene].sum()
    # Let's divide by the number of non-zero cells per cell-type to get average transcripts per cell (REMOVING THE CELLS WITH ZERO TRANSCRIPTS)
    nonzero_cell_count = (tissue_data_celltype.layers['counts'][:, tissue_data_celltype.var_names == gene] != 0).sum()
    average_count_per_cell[idx] = total_gene_count/nonzero_cell_count
    std_count_per_cell[idx] = np.std(tissue_data_celltype.layers['counts'][:, tissue_data_celltype.var_names == gene].toarray())

# Now let's make the dataframe for z-scored expression
if std_count_per_cell[idx] == 0:
    z_score_nonzero_df.loc[:, gene] = np.nan
else:
    z_score_nonzero_df = (raw_counts_per_cell_df_nonzero - average_count_per_cell) / std_count_per_cell

z_score_nonzero_df.to_csv('/Users/jacquelinechou/Downloads/slide2_r4_z_score_ligands_receptors_nonzero_cells.csv')

plt.figure(figsize=(8*2, 6))
sns.heatmap(z_score_nonzero_df.dropna().apply(pd.to_numeric, errors='coerce'), cmap='viridis', annot=True, fmt='.2f', )
plt.title('Z-scored Average Counts per cell (Slide2 Region4')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.savefig('/Users/jacquelinechou/Downloads/matrix_plot_slide2_region4_z_scored_nonzero_cells_matrixplot.png')

# Maybe I should be using the log-transformed and normalized values instead of raw counts. My gene panel is skewed towards specific celltypes

def heatmap_plots(adata_object, tissue, include_zero = True, raw_data = True):
    # first subset to tissue of interest
    subset_ad = adata_object[adata_object.obs['tissue'] == tissue]
    for major_celltype in raw_counts_df.index.dropna():
        tissue_data_celltype = subset_ad[subset_ad.obs['celltype_tier1_merged'] == major_celltype]
        if raw_data == True:
            tissue_data_matrix = tissue_data_celltype.layers['counts']
        else:  # use the log-transformed and normalized data
            tissue_data_matrix = tissue_data_celltype.X
        for gene in raw_counts_df.columns:
            total_gene_count = tissue_data_matrix[:, tissue_data_celltype.var_names == gene].sum()
            raw_counts_df.loc[raw_counts_df.index == major_celltype, gene] = total_gene_count
            if include_zero == True:
                number_of_cells_of_type = tissue_data_celltype.shape[0]
                cmap = 'viridis'
                # print(number_of_cells_of_type)
                if number_of_cells_of_type == 0:
                    raw_counts_per_cell_df.loc[raw_counts_per_cell_df.index == major_celltype, gene] = np.nan
                else:
                    raw_counts_per_cell_df.loc[raw_counts_per_cell_df.index == major_celltype, gene] = (
                                total_gene_count / number_of_cells_of_type)
            else:
                nonzero_cell_count = (tissue_data_celltype.layers['counts'][:, tissue_data_celltype.var_names == gene] != 0).sum()
                cmap = ListedColormap([(0.75, 0.75, 0.75)] + sns.color_palette("viridis", as_cmap=True).colors)
                if nonzero_cell_count == 0:
                    raw_counts_per_cell_df.loc[raw_counts_per_cell_df.index == major_celltype, gene] = np.nan
                else:
                    raw_counts_per_cell_df.loc[raw_counts_per_cell_df.index == major_celltype, gene] = (
                                total_gene_count / nonzero_cell_count)

    raw_counts_per_cell_df.to_csv(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/{tissue}_raw_counts_{raw_data}_per_cell_include_zero_{include_zero}.csv')
    plt.figure(figsize=(8 * 2, 6))
    # the np.nan will be changed to -1 and represented as a grey color in cmap
    sns.heatmap(raw_counts_per_cell_df.fillna(-1).apply(pd.to_numeric, errors='coerce'), cmap=cmap, annot=True, fmt='.2f', )
    plt.title(f'Average Number of Transcripts Per Cell ({tissue}) \n Include_zero: {include_zero}' )
    plt.xlabel('Ligands / Receptors')
    plt.ylabel('Major Cell-types')
    plt.savefig(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/matrix_plot_{tissue}_average_raw_counts_{raw_data}_matrixplot_zero_{include_zero}.png')

for tissue in cohort_data.obs['tissue'].unique():
    heatmap_plots(adata_object=cohort_data, tissue = tissue, include_zero= True, raw_data= True)

for tissue in cohort_data.obs['tissue'].unique():
    heatmap_plots(adata_object= cohort_data, tissue = tissue, include_zero= False)

def histogram_plots(adata_object, tissue, raw_data = True, genes = raw_counts_df.columns, row_order = raw_counts_df.index.dropna(), context = 'paper'):
    subset_ad = adata_object[adata_object.obs['tissue'] == tissue]
    if raw_data == True:
        subset_ad_exp_values = subset_ad[:, genes].layers['counts'].toarray()
        hist_bins = np.arange(0, np.max(subset_ad_exp_values) + 2) - 0.5
    else: # Use the log-transformed and median normalized counts
        subset_ad_exp_values = subset_ad[:, genes].X.toarray()
        hist_bins = None
    # let's create the dataframe needed to use seaborn's FacetGrid
    celltype_gene_df = pd.DataFrame(data=subset_ad_exp_values, columns=genes)
    celltype_gene_df['celltype'] = subset_ad.obs['celltype_tier1_merged'].values
    # print(f'{celltype_gene_df.shape}')
    print(f'{celltype_gene_df.head(10)}')
    celltype_gene_df = pd.melt(celltype_gene_df, id_vars=['celltype'], var_name='gene', value_name='expression')
    print(f'{celltype_gene_df.shape}')

    sns.set_context(context = context)
    fig = sns.FacetGrid(celltype_gene_df, col="gene", row="celltype", row_order = row_order)
    fig.map(plt.hist, 'expression', bins= hist_bins)
    for ax in fig.axes.flat:
        title = ax.get_title()
        gene, celltype = title.split(' | ')
        ax.set_title(f'{gene} \n {celltype}', fontsize = 10)
    plt.savefig(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/{tissue}_celltype_by_genes_histogram_raw_counts_{raw_data}.png')

# Plot histograms of transformed data
for tissue in cohort_data.obs['tissue'].unique():
    histogram_plots(cohort_data, tissue = tissue, raw_data = False)

# Plot histograms of raw counts
for tissue in cohort_data.obs['tissue'].unique():
    histogram_plots(cohort_data, tissue = tissue, raw_data = True)

# Including cells with zero counts
## Should I remove the zero entries? Since it's skewed towards zero? I can calculate both ways: with zeroes and without zeroes.
# Plot spatial plots of log-transformed / normalized data
# Let's plot across the tissue
def spatial_gene_plot(adata, tissue, all_cells = True,
                      # celltypes_interest,
                      raw_data = True, gene = 'IL1B'):
    if all_cells:
        # Subset to the specific tissue-of-interest
        subset_cohort_ad = adata[adata.obs['tissue'] == tissue]
    else:
        subset_cohort_ad = adata[adata.obs['celltype_tier1_merged'] == celltypes_interest]
    if raw_data:
        subset_cohort_ad_counts = subset_cohort_ad.layers['counts']
    else: # Use log transformed values
        subset_cohort_ad_counts = subset_cohort_ad.X
    plotting_factor = 300
    tissue_adata_y_max = np.ceil(np.max(subset_cohort_ad.obsm['spatial'][:, 1]))
    tissue_adata_y_min = np.floor(np.min(subset_cohort_ad.obsm['spatial'][:, 1]))
    tissue_adata_x_max = np.ceil(np.max(subset_cohort_ad.obsm['spatial'][:, 0]))
    tissue_adata_x_min = np.floor(np.min(subset_cohort_ad.obsm['spatial'][:, 1]))
    y_range = tissue_adata_y_max - tissue_adata_y_min
    x_range = tissue_adata_x_max - tissue_adata_x_min
    plt.figure(figsize=(x_range / plotting_factor, y_range / plotting_factor), dpi=300)
    # show cells in tissue as grey centroids
    plt.scatter(x=subset_cohort_ad.obsm['spatial'][:, 0], y=subset_cohort_ad.obsm['spatial'][:, 1], c='grey', s=1)
    # overlay with dots for transcript count
    plt.scatter(x=np.array(subset_cohort_ad.obsm['spatial'][:, 0]),
                y=np.array(subset_cohort_ad.obsm['spatial'][:, 1]),
                s=1, cmap='inferno_r',
                c=(subset_cohort_ad_counts[:, subset_cohort_ad.var_names == gene]).toarray().reshape(-1))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')
    plt.title(f'{gene} expression in {tissue} \n raw_data: {raw_data}')
    plt.colorbar(shrink=0.5)
    plt.savefig(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/{tissue}_{gene}_expression_all_cells_raw_data_{raw_data}.png',
                bbox_inches='tight')
    # plt.show()


# Plot spatial plots of raw data