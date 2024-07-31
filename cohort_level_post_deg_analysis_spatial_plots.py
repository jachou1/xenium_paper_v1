import scanpy as sc
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os
import glob

def plot_spatial_proximity(adata, output_dir):
    tissue_name = adata.obs['tissue'].unique()[0]
    celltypes_interest = ['Endothelial cells', 'Lining fibroblasts', 'Sublining fibroblasts', 'Myeloid cells', 'T-cells', 'Non-plasma B-cells', 'Plasma cells']
    for spatial_domains_idx in adata.obs['spatial_neighborhood_k_5'].unique():
        # print(spatial_domains_idx)
        for celltype in celltypes_interest:
            # plot the cells of a specific type that are proximal to this spatial domain
            celltype_proximal_cells = adata[adata.obs[f'celltype_proximity_sp{spatial_domains_idx}'] == f'{celltype}_proximal_sp{spatial_domains_idx}']
            n_proximal_cells = celltype_proximal_cells.shape[0]
            plt.figure(figsize = (20, 10), dpi = 200)
            plt.scatter(x = celltype_proximal_cells.obsm['spatial'][:, 0], y = celltype_proximal_cells.obsm['spatial'][:, 1], c = 'blue', s = 1, label = f'proximal {celltype}: {n_proximal_cells} cells')
            # plot the cells that are distal to this spatial domain
            celltype_distal_cells = adata[adata.obs[f'celltype_proximity_sp{spatial_domains_idx}'] == f'{celltype}_distal_sp{spatial_domains_idx}']
            n_distal_cells = celltype_distal_cells.shape[0]
            plt.scatter(x = celltype_distal_cells.obsm['spatial'][:, 0], y = celltype_distal_cells.obsm['spatial'][:, 1], c = 'orange', s =1, label = f'distal {celltype}: {n_distal_cells} cells')
            spatial_domain_cells = adata[adata.obs['spatial_neighborhood_k_5'] == spatial_domains_idx]
            plt.scatter(x = spatial_domain_cells.obsm['spatial'][:, 0], y = spatial_domain_cells.obsm['spatial'][:, 1], c = 'lightgrey', s = 1, label = f'spatial domain: {spatial_domains_idx}')
            plt.legend(loc = (1.1, 0.8), markerscale = 4)
            plt.gca().invert_yaxis()
            plt.gca().set_aspect('equal')
            plt.title(f'{celltype} used in DEG analysis for spatial domain {spatial_domains_idx}:')
            output_filename = os.path.join(output_dir,
                                           f'{tissue_name}_sp{spatial_domains_idx}_{celltype}_spatial_plot.png')
            plt.savefig(output_filename)
            plt.close()

def main(input_dir, output_dir):
    # Find all .h5ad files in the input_path directory
    h5ad_files = glob.glob(os.path.join(input_dir, '*.h5ad'))

    for h5ad_file in h5ad_files:
        adata = sc.read(h5ad_file)

        # Create a subdirectory for each AnnData file based on its name
        base_name = os.path.basename(h5ad_file).replace('.h5ad', '')
        sub_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(sub_output_dir, exist_ok=True)

        plot_spatial_proximity(adata, sub_output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate plots from processed AnnData objects.')
    parser.add_argument('input_dir', help='Path to the input AnnData files')
    parser.add_argument('output_dir', help='Directory to save the plots')

    args = parser.parse_args()
    # Print the arguments to verify their values
    # print(f'Input directory: {args.input_dir}')
    # print(f'Output directory: {args.output_dir}')
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    main(args.input_dir, args.output_dir)