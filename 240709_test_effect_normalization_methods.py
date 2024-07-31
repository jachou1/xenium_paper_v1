import scanpy as sc
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

cohort_adata = sc.read_h5ad('/Users/jacquelinechou/Downloads/240607_cohort_tcell_recluster_reindexed.h5ad')

# Let's look at the distribution of number of transcripts per cell
boxplot_settings = {
    'flierprops': dict(marker = 'o', color = 'red', markersize = 4, alpha = 0.2),
    'whiskerprops': dict(color= 'blue'),
    'capprops': dict(color = 'blue')
}
cohort_adata.obs.boxplot(column = 'n_counts', by = 'tissue', grid = False, vert = False, **boxplot_settings)
plt.title('Boxplot of n_counts_per_cell by tissue')
plt.suptitle('')
plt.tight_layout()
plt.savefig('/Users/jacquelinechou/Downloads/cohort_level_analysis/ncounts_per_cell_tissues.png')
plt.show()

# Let's look at each celltype in each tissue (n_counts)
tissues = np.unique(cohort_adata.obs['tissue'])

for tissue_i in tissues:
    tissue_ad = cohort_adata[cohort_adata.obs['tissue'] == tissue_i]
    tissue_ad.obs.boxplot(column='n_counts', by='celltype_tier1_merged', grid=False, vert=False, **boxplot_settings)
    plt.title(f'Boxplot of n_counts_per_cell by tissue: {tissue_i}')
    plt.suptitle('')
    plt.tight_layout()
    plt.xlim(0, 500)
    plt.savefig(f'/Users/jacquelinechou/Downloads/cohort_level_analysis/ncounts_per_cell_by_celltype_{tissue_i}.png')
    plt.show()

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

for i, major_celltypes in enumerate(list_of_markers):
    # let's look at the gene expression of each list of gene markers
    # indices = [np.where(cohort_adata.var_names.values == marker_gene)[0][0] if marker_gene in cohort_adata.var_names.values else None
    #            for marker_gene in major_celltypes]
    # Let's include cells that do not express these markers per cell-type
    # Let's subset the adata object to the genes of interest
    subset_ad_celltype_markers = cohort_adata[:, major_celltypes]
    # Let's further subset to celltype of interest; MAY WANT TO REVISIT TO SEE EXPRESSION OF THESE GENES IN ALL CELL-TYPES
    subset_ad_celltype_markers = subset_ad_celltype_markers[subset_ad_celltype_markers.obs['celltype_tier1_merged'] == celltypes[i]]
    counts_matrix = subset_ad_celltype_markers.layers['counts']
    counts_df = pd.DataFrame(counts_matrix.toarray(), columns = major_celltypes, index = subset_ad_celltype_markers.obs_names)
    # print(counts_df.head())
    combined_df = pd.concat([counts_df, subset_ad_celltype_markers.obs['tissue']], axis = 1)
    # print(combined_df.columns)
    # print(combined_df.head())
    # Then create a long-form dataframe that has the per-cell gene counts and tissue labels
    longform_df =pd.melt(combined_df, id_vars= ['tissue'], value_vars = major_celltypes, var_name='gene', value_name='expression')
    # print(longform_df.head())
    # Plotting strip plots per tcell_marker_module (group of genes) per tissue
    sns.stripplot(data=longform_df, x="gene", y="expression", hue="tissue", dodge=True, size = 3,
                  jitter = False, alpha = 0.2)

    # Customize the plot
    plt.xticks(rotation=90)
    plt.xlabel('Gene')
    plt.ylabel('Expression (raw counts)')
    plt.title(f'Stripplot of Marker Genes for {celltypes[i]}')
    plt.legend(title='Tissue', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/Users/jacquelinechou/Downloads/raw_counts_marker_genes_{celltypes[i]}_no_jitter.png')
    plt.show()

for i, major_celltypes in enumerate(list_of_markers):
    # let's look at the gene expression of each list of gene markers
    # indices = [np.where(cohort_adata.var_names.values == marker_gene)[0][0] if marker_gene in cohort_adata.var_names.values else None
    #            for marker_gene in major_celltypes]
    # Let's include cells that do not express these markers per cell-type
    # Let's subset the adata object to the genes of interest
    subset_ad_celltype_markers = cohort_adata[:, major_celltypes]
    # Let's further subset to celltype of interest; MAY WANT TO REVISIT TO SEE EXPRESSION OF THESE GENES IN ALL CELL-TYPES
    subset_ad_celltype_markers = subset_ad_celltype_markers[subset_ad_celltype_markers.obs['celltype_tier1_merged'] == celltypes[i]]
    counts_matrix = subset_ad_celltype_markers.layers['counts']
    counts_df = pd.DataFrame(counts_matrix.toarray(), columns = major_celltypes, index = subset_ad_celltype_markers.obs_names)
    # print(counts_df.head())
    combined_df = pd.concat([counts_df, subset_ad_celltype_markers.obs['tissue']], axis = 1)
    # print(combined_df.columns)
    # print(combined_df.head())
    # Then create a long-form dataframe that has the per-cell gene counts and tissue labels
    longform_df =pd.melt(combined_df, id_vars= ['tissue'], value_vars = major_celltypes, var_name='gene', value_name='expression')
    # print(longform_df.head())
    # If you want to plot the raw counts per gene separately, let's do the following:
    # for marker in major_celltypes:
    #     # filter to marker gene of interest, then plot
    #     longform_df_subset = longform_df[longform_df.loc[:, 'gene'] == marker]
    #     sns.violinplot(data=longform_df_subset, x="gene", y="expression", hue="tissue", width=1.0)
    #     # Customize the plot
    #     plt.xticks(rotation=90)
    #     plt.xlabel('Gene')
    #     plt.ylabel('Expression (raw counts)')
    #     plt.title(f'Violin plot of Marker Genes for {celltypes[i]}')
    #     plt.legend(title='Tissue', bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.tight_layout()
    #     plt.savefig(f'/Users/jacquelinechou/Downloads/raw_counts_{marker}_{celltypes[i]}_violin.png')
    #     plt.show()
        # Plotting strip plots per tcell_marker_module (group of genes) per tissue
    sns.violinplot(data=longform_df, x="gene", y="expression", hue="tissue", inner = None)

    # Customize the plot
    plt.xticks(rotation=90)
    plt.xlabel('Gene')
    plt.ylabel('Expression (raw counts)')
    plt.title(f'Violin plot of Marker Genes for {celltypes[i]}')
    plt.legend(title='Tissue', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/Users/jacquelinechou/Downloads/raw_counts_marker_genes_{celltypes[i]}_violin.png')
    plt.show()

# Let's make a swarmplot instead: why do swarmplots take soo long?

# Make a histogram like Tom suggested
for i, major_celltypes in enumerate(list_of_markers):
    # Let's subset the adata object to the genes of interest
    subset_ad_celltype_markers = cohort_adata[:, major_celltypes]
    # Let's further subset to celltype of interest; MAY WANT TO REVISIT TO SEE EXPRESSION OF THESE GENES IN ALL CELL-TYPES
    subset_ad_celltype_markers = subset_ad_celltype_markers[
        subset_ad_celltype_markers.obs['celltype_tier1_merged'] == celltypes[i]]
    counts_matrix = subset_ad_celltype_markers.layers['counts']
    counts_df = pd.DataFrame(counts_matrix.toarray(), columns=major_celltypes,
                             index=subset_ad_celltype_markers.obs_names)
    # print(counts_df.head())
    combined_df = pd.concat([counts_df, subset_ad_celltype_markers.obs['tissue']], axis=1)
    # print(combined_df.columns)
    # print(combined_df.head())
    # Then create a long-form dataframe that has the per-cell gene counts and tissue labels
    longform_df = pd.melt(combined_df, id_vars=['tissue'], value_vars=major_celltypes, var_name='gene',
                          value_name='expression')
    print(f'longform_df.shape: {longform_df.shape}')
    print(f'longform_df columns: {longform_df.columns}')
    for gene in major_celltypes:
        # print(f'gene: {gene}')
        subset_df = longform_df[longform_df['gene'] == gene]
        # print(longform_df.head())
        # Creates bins that span from x-0.5 and x + 0.5, so discrete numbers are captured in the bins
        bins = np.arange(0, max(subset_df.loc[:, 'expression']) + 2) - 0.5
        plt.figure(figsize=(10, 10))
        sns.histplot(data=subset_df, x="expression", hue="tissue", element='step', kde = True)

        # Customize the plot
        plt.ylabel('Number of cells')
        plt.xlabel('Expression (raw counts)')
        plt.title(f'Hist plot of {gene} for {celltypes[i]}')
        # plt.legend(title='Tissue', loc='upper right')
        plt.tight_layout()
        plt.savefig(f'/Users/jacquelinechou/Downloads/raw_counts_marker_{gene}_{celltypes[i]}_histplot_w_kde.png')

# Exercise: Assuming that normalization across tissues is not needed, let's see what the transcript level b/w any 2 tissues is
# Slide1_R5, Slide2_Region1

# Naive Approach
# DEG analysis of all sublining fibroblasts in different tissues
tissues = cohort_adata.obs['tissue'].unique()
tissues = tissues[tissues != 'slide2_r1']

# Let's create a column with the tissue_celltype
cohort_adata.obs['tissue_celltype'] = cohort_adata.obs['tissue'].str.cat(cohort_adata.obs['celltype_tier1_merged'], sep = '_')
celltype_interest = 'Sublining fibroblasts'

# for tissue in tissues:
group_of_tissues_to_analyze = [tissue + '_' + celltype_interest for tissue in tissues]
sc.tl.rank_genes_groups(cohort_adata, groupby = 'tissue_celltype', groups = group_of_tissues_to_analyze,
                    reference = 'slide2_r1_Sublining fibroblasts', method = 'wilcoxon', key_added = 'tissue_sublining_fibs_degs')

sc.pl.rank_genes_groups_dotplot(
    cohort_adata, groupby = 'tissue_celltype', standard_scale="var", n_genes=15, key = 'tissue_sublining_fibs_degs',
    save = 'tissues_sublining_fibs_degs_dotplot.png')

# Create mask for cell-types of interest for plotting [0 v. 1 per celltype]
mask = slide1_region3_tissue.obs['celltype_tier1_proximal_tb'].isin([f'{celltype}1', f'{celltype}0'])
top_5_genes = pd.DataFrame(slide1_region3_tissue.uns[f'{celltype}_1_ref_0_degs']['names']).head(5)
genes_to_plot = top_5_genes.values.flatten()
# Let's make the violin plots for each celltype:
sc.pl.violin(slide1_region3_tissue[mask], keys = genes_to_plot, groupby = 'celltype_tier1_proximal_tb', save = f's1_r3_{celltype}_tb_1_0.png')

# Let's get the CSV file of lining fibroblasts' DEGs:
sublining_fibs_tissue_degs = sc.get.rank_genes_groups_df(cohort_adata, key = 'tissue_sublining_fibs_degs', pval_cutoff= 0.05,
                                                         group = None)

sublining_fibs_tissue_degs.to_csv(f'/Users/jacquelinechou/Downloads/across_tissues_sublining_fibs_degs.csv')

