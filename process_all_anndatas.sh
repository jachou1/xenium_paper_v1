#!/bin/bash

base_dir="/Users/jacquelinechou/Downloads"

# List of input AnnData files
subfolders=("s2_r1" "s2_r2" "s2_r3" "s2_r4" "s2_r5" "s2_r6")  # Add all your file names here

# Output directory
output_dir="/Users/jacquelinechou/Downloads/processed_files"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Initialize an array to hold the input file paths
input_files=()

# Collect all .h5ad files from the specified subfolders
for subfolder in "${subfolders[@]}"; do
    # Construct the full path to the subfolder
    full_subfolder_path="$base_dir/$subfolder"
    echo "The full subfolder path is $full_subfolder_path"
    # Check if the subfolder exists
    if [ -d "$full_subfolder_path" ]; then
        echo "Processing subfolder: $full_subfolder_path"
        for file in "$full_subfolder_path"/*.h5ad; do
            # Check if there are any .h5ad files
            if [ -e "$file" ] && [ ! -d "$file" ]; then
                input_files+=("$file")
            fi
        done
    else
        echo "Subfolder $full_subfolder_path does not exist."
    fi
done

# Check if input_files array is empty
if [ ${#input_files[@]} -eq 0 ]; then
    echo "No .h5ad files found in the specified directories."
    exit 1
fi

# Run the Python script with all input files
python /Users/jacquelinechou/Code/scanpy_env/cohort_level_spatial_domain_deg_analysis.py "${input_files[@]}" "$output_dir"
