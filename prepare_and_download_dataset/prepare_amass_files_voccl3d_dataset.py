import argparse
import os
import numpy as np
import shutil
import glob

def main():
    parser = argparse.ArgumentParser(description="Prepare the required amass files for VOccl3D-dataset")
    parser.add_argument("--amass_file_path", type=str, required=True, help=".npy file for match pairs of amass")
    parser.add_argument("--root_amass_dir", type=str, required=True, help="Give the path of Directory where AMASS dataset is downloaded")
    parser.add_argument("--root_voccl3d_dir", type=str, required=True, help="Directory of downloaded VOccl3D dataset")
    args = parser.parse_args()

    amass_file = args.amass_file_path
    amass_root_dir = args.root_amass_dir
    voccl3d_root_dir = args.root_voccl3d_dir

    amass_match_dict = np.load(amass_file, allow_pickle=True).item()
    amass_match_dict = {'/'.join(key.split('/')[1:]):value for key,value in amass_match_dict.items()}
    all_files_voccl3d_dataset = glob.glob(voccl3d_root_dir + '/*/transformation_files/*.npy')
    for file in all_files_voccl3d_dataset:
        print(file)

        dst_file = (file.split('/')[-1]).replace('.npy', '.npz')
        # breakpoint()
        dst_dir = os.path.dirname(file.replace('transformation_files', 'amass_files'))
        os.makedirs(dst_dir, exist_ok=True)

        match_key = os.path.join(*dst_dir.split('/')[-2:], dst_file)
        src_file = amass_match_dict[match_key].split('/')[1:]
        src_file = os.path.join(amass_root_dir, *src_file)

        shutil.copy(src_file, os.path.join(dst_dir, dst_file))


if __name__ == '__main__':
    main()

# syntax code: python prepare_amass_files_voccl3d_dataset.py --amass_file_path /Yash/occl_dataset/voccl3d_dataset_release_amass_file_pairs.npy --root_amass_dir /Yash/AMASS --root_voccl3d_dir /Yash/occl_dataset/voccl3d_dataset_release_tiny