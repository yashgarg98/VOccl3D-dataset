# VOccl3D-dataset
This repository contains information about VOccl3D dataset and evaluation codes 

## Setup the conda environment
Follow the installation steps from BEDLAM paper's repository (link: https://github.com/pixelite1201/BEDLAM#install) 

## Download and prepare VOccl3D dataset
Follow the 'prepare_and_download_dataset\README.md' file complete this step.

## Visualize the Mesh and 3D keypoints of our dataset
First download and prepare the VOccl3D dataset. Run the script *'visualize_voccl3d_dataset.py'* to visualize the ground-truth mesh overlay and 3D keypoints.

```bash
python visualize_voccl3d_dataset.py \
--voccl3d_video_folder /voccl3d_dataset_release_tiny/scene2_view1/images/SMPLX-female_ulr3_stageii \
--ground_truth_scene_npy /voccl3d_dataset_release_tiny/scene2_view1/scene2_view1_ground_truth_labels.npy \
--helper_folder_path /voccl3d_dataset_release_helper_files \
--vis_mesh \
--vis_keypoint
```