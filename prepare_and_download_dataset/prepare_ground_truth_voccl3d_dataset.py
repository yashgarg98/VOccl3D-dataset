import numpy as np
import os
import pickle
import torch
import re 
from tqdm import tqdm
import smplx
import argparse

# def get_faces(helper_folder_path):
#     smplx_model = smplx.create(
#                 os.path.join(helper_folder_path, 'models_smplx'),
#                 model_type="smplx",
#                 gender='neutral',
#                 use_face_contour=False,
#                 num_betas=10,
#                 num_expression_coeffs=10,
#                 ext="npz",
#                 use_pca=False,  # explicitly control hand pose
#                 flat_hand_mean=True,  # use a flatten hand default pose
#             )
#     faces = smplx_model.faces.astype(np.float32)
#     return faces

def parse_matrix(lines, start_index, rows, cols):
    matrix = []
    for i in range(rows):
        row = lines[start_index + i].strip('[]\n').split(',')
        matrix.append([float(x) for x in row])
    return np.array(matrix)

def get_smplx_vertices_and_joints(smpl_model_male, smpl_model_female, smpl_model_neutral, global_orient, pose, betas, gender):
    if gender=='male':
        model_out = smpl_model_male(betas=torch.tensor(betas[:11]).unsqueeze(0).float(),
                                    global_orient=torch.tensor(global_orient).float(),
                                    body_pose=torch.tensor(pose[3:66]).unsqueeze(0).float()
                                    )
    elif gender=='female':
        model_out = smpl_model_female(betas=torch.tensor(betas[:11]).unsqueeze(0).float(),
                                    global_orient=torch.tensor(global_orient).float(),
                                    body_pose=torch.tensor(pose[3:66]).unsqueeze(0).float()
                                    )
    elif gender=='neutral':
        model_out = smpl_model_neutral(betas=torch.tensor(betas[:11]).unsqueeze(0).float(),
                                    global_orient=torch.tensor(global_orient).float(),
                                    body_pose=torch.tensor(pose[3:66]).unsqueeze(0).float()
                                    )
    else:
        print(f"Incorrect gender, provided {gender}")

    vertices = model_out.vertices.squeeze(0).detach().cpu().numpy()
    joints = model_out.joints.squeeze(0).detach().cpu().numpy()
    return vertices, joints

def get_project2D(joints_3d, translation, camera_intrinsic):
    joints_3d = joints_3d + translation
    joints_3d = torch.tensor(joints_3d).float()
    camera_intrinsic = torch.tensor(camera_intrinsic).float()

    projected_points = joints_3d / joints_3d[:, -1].unsqueeze(-1)
    projected_points = torch.einsum('ij, kj->ki', camera_intrinsic, projected_points.float())

    return projected_points.detach().cpu().numpy()

def get_bbox_valid(joints, img_height, img_width, rescale):
    #Get bbox using keypoints
    valid_j = []
    joints = np.copy(joints)
    for j in joints:
        if j[0] > img_width or j[1] > img_height or j[0] < 0 or j[1] < 0:
            continue
        else:
            valid_j.append(j)

    if len(valid_j) < 1:
        return [-1, -1], -1, len(valid_j), [-1, -1, -1, -1]

    joints = np.array(valid_j)

    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]

    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

    scale *= rescale
    return center, scale, len(valid_j), bbox

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser(description="Prepare the ground-truth files for VOccl3D-dataset")
    parser.add_argument("--helper_folder_path", type=str, required=True, help="helper files to compute the ground-truth")
    parser.add_argument("--root_voccl3d_dir", type=str, required=True, help="Directory of downloaded VOccl3D dataset")
    args = parser.parse_args()

    helper_folder_path = args.helper_folder_path
    root_voccl3d_dir = args.root_voccl3d_dir
    
    MODEL_FOLDER = os.path.join(helper_folder_path, 'models')  #
    smpl_model_male = smplx.create(MODEL_FOLDER,
                                    model_type = 'smplx',
                                    gender='male',
                                    ext='npz',
                                    num_betas=11)

    smpl_model_female = smplx.create(MODEL_FOLDER,
                                    model_type = 'smplx',
                                    gender='female',
                                    ext='npz',
                                    num_betas=11)

    smpl_model_neutral = smplx.create(MODEL_FOLDER,
                                    model_type = 'smplx',
                                    gender='neutral',
                                    ext='npz',
                                    num_betas=11)

    smplx2smpl = pickle.load(open(os.path.join(helper_folder_path, 'smplx2smpl.pkl'), 'rb'))
    smplx2smpl = torch.tensor(smplx2smpl['matrix'][None], dtype=torch.float32).to(device)
    J_regressor = torch.from_numpy(np.load(os.path.join(helper_folder_path, 'J_regressor_h36m.npy'))).float().to(device)
    batch_size=1
    J_regressor_batch_smpl = J_regressor[None, :].expand(batch_size, -1, -1)

    for scene in os.listdir(root_voccl3d_dir):
        root_dir = os.path.join(root_voccl3d_dir, scene)
        video_file_path = os.path.join(root_dir, 'images')

        camera_txt_file_path = os.path.join(video_file_path, 'camera_parameters.txt')
        with open(camera_txt_file_path, 'r') as f:
            lines = f.readlines()
        # Extract intrinsic matrix
        intrinsic_start = lines.index("Camera Intrinsic Matrix:\n") + 1
        camera_intrinsic = parse_matrix(lines, intrinsic_start, rows=3, cols=3)
        # Extract extrinsic matrix
        extrinsic_start = lines.index("Camera Extrinsic Matrix:\n") + 1
        world_to_camera_matrix = parse_matrix(lines, extrinsic_start, rows=4, cols=4)

        save_dict_npy = {}
        for folder_id, video_folder in enumerate(sorted(os.listdir(video_file_path))):
            if video_folder=='camera_parameters.txt':
                continue
            print(f'Folder name: {folder_id}, {video_folder}')
            video_dir_path = os.path.join(video_file_path, video_folder)

            ## load the transformation files
            transform_anim_path = video_dir_path.replace('images', 'transformation_files') + '.npy'
            transform_anim = np.load(transform_anim_path, allow_pickle=True).item()
            first_index = next(iter(transform_anim))

            img_files_name = sorted([img_name for img_name in os.listdir(video_dir_path) if img_name.endswith('.png')])

            ## reading original amass dataset
            amass_filepath = video_dir_path.replace('images', 'amass_files') + '.npz'
            orig_amass = np.load(amass_filepath, allow_pickle=True)
            amass_poses = orig_amass['poses']
            betas = orig_amass['betas']

            gt_single_npy = {}
            for idx, img_name in tqdm(enumerate(img_files_name)):
                gt_dict = {}
                img_int = int(img_name[4:-4])
                gender_str = video_dir_path.split('/')[-1]
                match = re.search(r'(male|female|neutral)', gender_str)
                if match:
                    gender = match.group(0)
                else:
                    gender = 'None'
                
                ## original data from AMASS
                poses = amass_poses[(first_index + idx -1)*4]

                # transformation data
                transform_data = transform_anim[first_index+idx]
                
                camera_verts_centroid = transform_data['camera_verts_centroid']
                global_orient_vert = transform_data['global_orient_vert']

                cam_vertices, cam_3djoints = get_smplx_vertices_and_joints(smpl_model_male, smpl_model_female, smpl_model_neutral, global_orient=global_orient_vert, pose=poses, betas=betas, gender=gender)
                cam_translation = camera_verts_centroid.reshape(1,-1) - np.mean(cam_vertices, axis=0).reshape(1,-1)
                cam_2djoints = get_project2D(cam_3djoints, cam_translation, camera_intrinsic)
                cam_vertices_trans = cam_vertices + cam_translation

                ## code to get the bbox and center,scale
                SCALE_FACTOR_BBOX = 1.2
                center, scale, num_vis_joints, bbox = get_bbox_valid(cam_2djoints[:22], rescale=SCALE_FACTOR_BBOX, img_width=700, img_height=700)

                # ## save the gt data
                corrected_poses = poses.copy()
                corrected_poses[:3] = global_orient_vert

                pred_cam_vertices = torch.matmul(smplx2smpl.repeat(batch_size, 1, 1).to(device), torch.from_numpy(cam_vertices_trans).float().to(device))
                joints3D = torch.matmul(J_regressor_batch_smpl, pred_cam_vertices).cpu().numpy()

                gt_dict['center'] = center
                gt_dict['scale'] = scale
                gt_dict['global_orient_vert'] = global_orient_vert
                gt_dict['pose_cam'] = corrected_poses
                gt_dict['shape'] = betas[:11]
                gt_dict['cam_translation'] = cam_translation
                gt_dict['gender'] = gender
                gt_dict['gtkps'] = cam_2djoints
                gt_dict['camera_extrinsic'] = world_to_camera_matrix
                gt_dict['camera_intrinsic'] = camera_intrinsic
                gt_dict['bbox'] = bbox
                gt_dict['cam_vertices_raw'] = cam_vertices_trans
                gt_dict['joints3D'] = joints3D

                gt_single_npy[img_name] = gt_dict
            save_folder_name = os.path.join('/'.join(video_file_path.split('/')[-2:]), video_folder)
            save_dict_npy[save_folder_name] = gt_single_npy

        save_dir = os.path.join(root_dir, root_dir.split('/')[-1] + '_ground_truth_labels.npy')
        breakpoint()
        np.save(save_dir, save_dict_npy)

if __name__ == '__main__':
    main()