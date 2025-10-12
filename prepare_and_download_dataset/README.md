## Follow this script to download and prepare the VOccl3D dataset
Steps:  
1. Follow the instructions of *"prepare_dataset\prepare_AMASS.md"* to prepare AMASS dataset.
2. Run the command below to download the complete VOccl3D dataset at any desired directory (provide the same dir path as ROOT_DIR at step 4).

```bash
wget --recursive --no-host-directories --no-parent  --cut-dirs=1 https://data.ucr.edu/datasets/voccl3d_dataset_release/
```

3. Prepare the utils or helper data. Run the script *'prepare_and_download_dataset\download_helper_files.sh'*. Before running the bash file please register on the [SMPL-X](https://smpl-x.is.tue.mpg.de) and [BEDLAM](https://bedlam.is.tue.mpg.de/). (provide the same dir path as helper_file_dir at step 4)

4. '*cd prepare_and_download_dataset*', then run the bash file *"prepare_voccl3d_dataset.sh"* to prepare the annotations for the downloaded VOccl3D dataset. Edit the following inside the *"prepare_voccl3d_dataset.sh"* file before running.

    * *ROOT_DIR* (line3): Update the path to be the same as downloaded VOccl3D dataset directory after step2.
    *  *amass_name_pair_file* (line4): Update the correct path of *"prepare_and_download_dataset\voccl3d_dataset_release_amass_file_pairs.npy"*.
    * *helper_file_dir* (line5): 
    * *original_amass* (line6): Provide the downloaded directory of AMASS dataset (follow step 1).

 
 **Please make sure the file path are correct, otherwise it can not succeed.**