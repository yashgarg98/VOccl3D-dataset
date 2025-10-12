#!/bin/bash

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X model
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p voccl3d_dataset_release_helper_files/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O './voccl3d_dataset_release_helper_files/smplx.zip' --no-check-certificate --continue
unzip voccl3d_dataset_release_helper_files/smplx.zip -d voccl3d_dataset_release_helper_files/smplx

mv /data/AmitRoyChowdhury/Yash/occl_dataset/voccl3d_dataset_release_helper_files/smplx/models voccl3d_dataset_release_helper_files/

rm -rf voccl3d_dataset_release_helper_files/smplx

# utils
echo -e "\nYou need to register at https://bedlam.is.tue.mpg.de/"
read -p "Username (BEDLAM):" username
read -p "Password (BEDLAM):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=bedlam&resume=1&sfile=utils.zip' -O './voccl3d_dataset_release_helper_files/utils.zip' --no-check-certificate --continue
unzip voccl3d_dataset_release_helper_files/utils.zip -d voccl3d_dataset_release_helper_files

mv voccl3d_dataset_release_helper_files/utils/J_regressor_h36m.npy voccl3d_dataset_release_helper_files/
mv voccl3d_dataset_release_helper_files/utils/smplx2smpl.pkl voccl3d_dataset_release_helper_files/

rm -rf voccl3d_dataset_release_helper_files/utils

echo -e "\nDownloaded helper files are in voccl3d_dataset_release_helper_files/"