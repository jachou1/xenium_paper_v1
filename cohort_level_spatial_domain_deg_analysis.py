import scanpy as sc
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os

def has_cells(adata, groupby, group):
    """
    Check if a group has any cells in the AnnData object.

    Parameters:
    - adata: AnnData object containing single-cell data.
    - groupby: The column in adata.obs to group cells by.
    - group: The specific group to check.

    Returns:
    - True if the group has cells, False otherwise.
    """
    return np.sum(adata.obs[groupby] == group) > 0
def process_anndata(adata, output_path):
    tissue_name = adata.obs['tissue'].unique()[0]
    # Initialize new columns for proximal/distal information
    for i in range(5):
        adata.obs[f'proximal_sp{i}'] = None

    # Detect proximal cells to each spatial domain
    for spatial_domain_i in adata.obs['spatial_neighborhood_k_5'].unique():
        spatial_domain_cells = adata.obsm['spatial'][adata.obs['spatial_neighborhood_k_5'] == spatial_domain_i, :]
        all_cells = adata.obsm['spatial']
        kdtree = KDTree(all_cells)
        distances_threshold = 30
        proximal_cells_spatial_domain = kdtree.query_ball_point(spatial_domain_cells, r=distances_threshold)
        proximal_cells_spatial_domain_indices = np.unique(np.concatenate(proximal_cells_spatial_domain))

        # Assume all cells are distal initially
        adata.obs[f'proximal_sp{spatial_domain_i}'] = f'distal_sp{spatial_domain_i}'
        adata.obs.iloc[proximal_cells_spatial_domain_indices, adata.obs.columns.get_loc(
            f'proximal_sp{spatial_domain_i}')] = f'proximal_sp{spatial_domain_i}'

        # Plot to confirm proximal cells
        plt.figure(figsize=(20, 10), dpi=100)
        proximal_x_coords = adata.obsm['spatial'][proximal_cells_spatial_domain_indices, 0]
        proximal_y_coords = adata.obsm['spatial'][proximal_cells_spatial_domain_indices, 1]
        plt.scatter(x=proximal_x_coords, y=proximal_y_coords, s=1, c='blue', label = 'proximal cells')
        plt.scatter(x=spatial_domain_cells[:, 0], y=spatial_domain_cells[:, 1], s=1, c='orange', label = 'spatial domain')
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.title(f'{tissue_name} Spatial domain: {spatial_domain_i}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{tissue_name}_spatial_domain_{spatial_domain_i}_kdtree_30_microns_proximal_cells.png'))
        plt.close() # Close plot to avoid memory issues

    # Create concatenated columns for cell type proximity
    for spatial_domain_i in adata.obs['spatial_neighborhood_k_5'].unique():
        adata.obs[f'celltype_proximity_sp{spatial_domain_i}'] = adata.obs['celltype_tier1_merged'].str.cat(
            adata.obs[f'proximal_sp{spatial_domain_i}'], sep='_')

    # Differential expression analysis
    celltypes_interest = ['Myeloid cells', 'T-cells', 'Non-plasma B-cells', 'Lining fibroblasts',
                          'Sublining fibroblasts', 'Endothelial cells', 'Plasma cells']

    for spatial_domains_idx in adata.obs['spatial_neighborhood_k_5'].unique():
        for celltype in celltypes_interest:
            proximal_group = f'{celltype}_proximal_sp{spatial_domains_idx}'
            distal_group = f'{celltype}_distal_sp{spatial_domains_idx}'

            # Check if either group has zero cells
            if not has_cells(adata, f'celltype_proximity_sp{spatial_domains_idx}', proximal_group) or \
                    not has_cells(adata, f'celltype_proximity_sp{spatial_domains_idx}', distal_group):
                print(f"Skipping comparison for {celltype} in spatial domain {spatial_domains_idx} due to zero cells.")
                continue

            try:
                sc.tl.rank_genes_groups(adata, groupby=f'celltype_proximity_sp{spatial_domains_idx}',
                                        groups=[f'{celltype}_proximal_sp{spatial_domains_idx}'],
                                        reference=f'{celltype}_distal_sp{spatial_domains_idx}',
                                        method='wilcoxon', key_added=f'{celltype}_sp_{spatial_domains_idx}_1_ref_0_degs')
                sc.pl.rank_genes_groups_dotplot(adata, groupby=f'celltype_proximity_sp{spatial_domains_idx}',
                                                standard_scale="var", n_genes=15,
                                                key=f'{celltype}_sp_{spatial_domains_idx}_1_ref_0_degs',
                                                save=f'{tissue_name}_{celltype}_in_sp_{spatial_domains_idx}_vs_out_dotplot.png',
                                                show = False)
                print('done plotting dotplot')
                celltype_sp_degs = sc.get.rank_genes_groups_df(adata,
                                                               key=f'{celltype}_sp_{spatial_domains_idx}_1_ref_0_degs',
                                                               group=f'{celltype}_proximal_sp{spatial_domains_idx}',
                                                               pval_cutoff=0.05)
                print(f'df.shape: {celltype_sp_degs.shape}')
                celltype_sp_degs.to_csv(os.path.join(output_path, f'{tissue_name}_{celltype}_sp_{spatial_domains_idx}_degs.csv'))
            except Exception as e:
                print(
                    f"An error occurred during differential expression analysis for {celltype} in spatial domain {spatial_domains_idx}: {e}")

    return adata



def main(input_paths, output_dir):
    for input_path in input_paths:
        adata = sc.read(input_path)
        processed_adata = process_anndata(adata, output_dir)

        # Create the output file path
        base_name = os.path.basename(input_path).replace('.h5ad', '_processed.h5ad')
        output_path = os.path.join(output_dir, base_name)

        # Save the processed AnnData object
        processed_adata.write(output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process multiple AnnData objects.')
    parser.add_argument('input_paths', nargs='+', help='Paths to the input AnnData files')
    parser.add_argument('output_dir', help='Directory to save the processed AnnData files')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    main(args.input_paths, args.output_dir)
