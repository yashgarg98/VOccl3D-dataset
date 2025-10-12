#!/bin/bash
# Please update/check the ROOT_DIR, amass_name_pair_file, helper_file_dir, original_amass variables before running this script.
ROOT_DIR="/occl_dataset/voccl3d_dataset_release"
amass_name_pair_file="VOccl3D-dataset\prepare_and_download_dataset\voccl3d_dataset_release_amass_file_pairs.npy"
helper_file_dir="/occl_dataset/voccl3d_dataset_release_helper_files"
original_amass=".../AMASS"

# Iterate over subdirectories of root
for dir in "$ROOT_DIR"/*/; do
    # Remove trailing slash and get just the folder name
    some_folder=$(basename "$dir")
    echo "Processing: $some_folder"

    image_tar_path="$dir/images.tar.gz"
    transformation_tar_path="$dir/transformation_files.tar.gz"

    # If already extracted, skip this folder
    if [[ -d "$dir/images" && -d "$dir/transformation_files" ]]; then
        echo "  -> Already extracted, skipping $some_folder"
        continue
    fi

    tar -xzf "$image_tar_path" -C "$dir/"
    tar -xzf "$transformation_tar_path" -C "$dir/"
done

# Call python script to prepare AMASS files and put them into specified folders
python prepare_amass_files_voccl3d_dataset.py --amass_file_path "$amass_name_pair_file" --root_amass_dir "$original_amass" --root_voccl3d_dir "$ROOT_DIR"

# Call main ground-truth creation files to prepare our labels. Optional: parse arg to visualize overlay meshes, bounding boxes, 2D key-points.
python prepare_ground_truth_voccl3d_dataset.py --helper_folder_path "$helper_file_dir" --root_voccl3d_dir "$ROOT_DIR"

echo "---------------------------- dataset preparation finished ----------------------------"