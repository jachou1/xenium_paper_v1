import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import glob
import os


def concatenate_csv_files(csv_files, output_dir):
    """
    Read and concatenate CSV files into a dictionary of DataFrames by tissue and save the concatenated DataFrame.
    """
    tissue_dataframes = {}

    # Iterate through the CSV files
    for file in csv_files:
        # Read the CSV file
        df = pd.read_csv(file)

        # Check if the DataFrame is empty
        if df.empty:
            print(f"File {file} is empty and will be skipped.")
            continue

        # Extract the base filename without the directory and extension
        base_filename = os.path.basename(file)
        parts = base_filename.replace('.csv', '').split('_')
        tissue_name = parts[0]+'_'+parts[1]  # Extract tissue name from the filename
        celltype = parts[2]
        spatial_domain_value = parts[3] + '_' + parts[4]

        # Calculate the negative pval_adj
        df['nlog10_pval_adj'] = -np.log10(df.pvals_adj)
        # Create new columns based on extracted information
        df['Celltype_tested'] = celltype
        df['Spatial_domain'] = spatial_domain_value

        # Append the DataFrame to the list for this tissue
        if tissue_name not in tissue_dataframes:
            tissue_dataframes[tissue_name] = []
        tissue_dataframes[tissue_name].append(df)

    # Save concatenated DataFrames for each tissue to a CSV file
    for tissue_name, dataframes in tissue_dataframes.items():
        final_df = pd.concat(dataframes, ignore_index=True)
        output_csv_file = os.path.join(output_dir, f'{tissue_name}_spatial_domain_degs_concatenated.csv')
        final_df.to_csv(output_csv_file, index=False)
        print(f'Saved concatenated CSV for {tissue_name} as {output_csv_file}')

    return tissue_dataframes


def custom_volcano_plot(data, x, y, color, **kwargs):
    """
    Custom volcano plot function with annotations.
    """
    ax = plt.gca()
    sns.scatterplot(data=data, x=x, y=y, color=color, ax=ax, **kwargs)

    # Add horizontal and vertical lines
    ax.axhline(5, zorder=0, c='k', lw=2, ls='--')
    ax.axvline(-1, zorder=0, c='k', lw=2, ls='--')
    ax.axvline(1, zorder=0, c='k', lw=2, ls='--')

    # Add text annotations for significant points
    texts = []
    for i in range(len(data)):
        if data.iloc[i][y] > 5 and abs(data.iloc[i][x]) > 1:
            texts.append(
                plt.text(data.iloc[i][x], y=data.iloc[i][y], s=data.iloc[i]['names']))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k'))


def plot_facetgrid(tissue_dataframes, output_dir):
    """
    Generate FacetGrid plots for each tissue.
    """
    # Generate plots for each tissue
    for tissue_name, dataframes in tissue_dataframes.items():
        # Concatenate all DataFrames for this tissue
        # print(f'Processing tissue: {tissue_name}')
        final_df = pd.concat(dataframes, ignore_index=True)

        # Create the FacetGrid plot
        col_order = ['Endothelial cells', 'Lining fibroblasts', 'Sublining fibroblasts', 'Myeloid cells', 'T-cells',
                     'Non-plasma B-cells', 'Plasma cells']
        row_order = ['sp_0', 'sp_1', 'sp_2', 'sp_3', 'sp_4']
        figure_grid = sns.FacetGrid(final_df, col='Celltype_tested', row='Spatial_domain',
                                    row_order=row_order, col_order=col_order, margin_titles=True)

        # Map a volcano plot
        figure_grid.map_dataframe(custom_volcano_plot, x='logfoldchanges', y='nlog10_pval_adj', color= '#80afd6')

        figure_grid.set_axis_labels('Log(2)Fold Change', '-log10(p-value)')

        # Save the plot with the tissue information in the filename
        output_filename = os.path.join(output_dir, f'{tissue_name}_volcano_plot_spatial_domain_by_celltype.png')
        plt.savefig(output_filename)
        plt.close()  # Close plot to avoid memory issues


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate volcano plots from CSV files.')
    parser.add_argument('input_dir', help='Directory containing the CSV files')
    parser.add_argument('output_dir', help='Directory to save the plots and concatenated CSV files')

    args = parser.parse_args()

    # List all CSV files in the input directory
    csv_files = glob.glob(os.path.join(args.input_dir, '*.csv'))

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Concatenate CSV files and generate FacetGrid plots
    tissue_dataframes = concatenate_csv_files(csv_files, args.output_dir)
    plot_facetgrid(tissue_dataframes, args.output_dir)
