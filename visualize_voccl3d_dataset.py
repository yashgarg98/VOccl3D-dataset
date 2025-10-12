import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pickle
import torch
import re 
from tqdm import tqdm
import smplx
import cv2
import pyrender
import colorsys
import trimesh
import argparse
import matplotlib.pyplot as plt

def get_faces(helper_folder_path):
    smplx_model = smplx.create(
                os.path.join(helper_folder_path, 'models_smplx'),
                model_type="smplx",
                gender='neutral',
                use_face_contour=False,
                num_betas=10,
                num_expression_coeffs=10,
                ext="npz",
                use_pca=False,  # explicitly control hand pose
                flat_hand_mean=True,  # use a flatten hand default pose
            )
    faces = smplx_model.faces.astype(np.float32)
    return faces

def render_3d_object_on_image(vertices, faces, image_path, output_image_path, focal_length_x=600, focal_length_y=600, image_width=512, image_height=512, save_mask_path=None, save_bbox_path = None):
    # Load the background image
    background_image = cv2.imread(image_path)
    background_image_rgb = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    background_image_rgb1 = background_image_rgb.copy()

    scene = pyrender.Scene()
    mesh_color = colorsys.hsv_to_rgb(0.6, 0.5, 1.0)
    material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=mesh_color)
    # Set up the camera
    camera = pyrender.IntrinsicsCamera(fx=focal_length_x, fy=focal_length_y, cx=image_width / 2, cy=image_height / 2)
    camera_pose = np.eye(4)  # Camera at origin, no rotation or translation
    scene.add(camera, pose=camera_pose)

    # Add a light source
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)
    # Need to flip x-axis
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

    # Create a trimesh object from the vertices and faces
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.apply_transform(rot)
    # Convert the trimesh object to a pyrender mesh and add it to the scene
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
    # mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyrender)

    # Set up the renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

    # Render the scene
    color, depth = renderer.render(scene)

    # Overlay the rendered 3D object on the background image
    mask = depth > 0  # Only render where the object is
    background_image_rgb[mask] = color[mask]
    if save_mask_path is not None:
        mask_uint8 = (mask * 255).astype(np.uint8)
        image = Image.fromarray(mask_uint8)
        image.save(save_mask_path)

    # Save or display the resulting image
    cv2.imwrite(output_image_path, cv2.cvtColor(background_image_rgb, cv2.COLOR_RGB2BGR))

def visualize_2d(image_path, save_path, joints2d):
    img = cv2.imread(image_path)
    img = img[:, :, ::-1]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for i in range(len(joints2d)):
        ax.scatter(joints2d[i, 0], joints2d[i, 1], s=0.2)
        # Add the index as text next to the keypoint
        ax.text(joints2d[i, 0], joints2d[i, 1], str(i + 1), fontsize=2, color='#39FF14')
    plt.savefig(save_path)

def main():
    parser = argparse.ArgumentParser(description="visualize mesh and 2D-key-points for VOccl3D-dataset")
    parser.add_argument("--voccl3d_video_folder", type=str, required=True, help="video folder to perform the visualization")
    parser.add_argument("--ground_truth_scene_npy", type=str, required=True, help="ground-truth npy file of the corresponding scene")
    parser.add_argument("--helper_folder_path", type=str, required=True, help="helper files to compute the ground-truth")
    parser.add_argument("--vis_mesh", action="store_true", help="To perform mesh visualization")
    parser.add_argument("--vis_keypoint", action="store_true", help="To perform 2D key-point visualization")
    parser.add_argument("--save_visualize_dir", type=str, default=False, help="helper files to compute the ground-truth")
    args = parser.parse_args()

    video_dir = args.voccl3d_video_folder
    helper_folder_path = args.helper_folder_path
    ground_truth = np.load(args.ground_truth_scene_npy, allow_pickle=True).item()
    get_key = [key for key in ground_truth.keys() if video_dir.split('/')[-1] in key][0]
    ground_truth = ground_truth[get_key]

    if args.save_visualize_dir:
        os.makedirs(args.save_visualize_dir, exist_ok=True)
        save_folder_dir = args.save_visualize_dir
        save_folder = os.path.join(save_folder_dir, video_dir.split('/')[-1] + '_visualize')
        os.makedirs(save_folder, exist_ok=True)
    else:
        folder_name = video_dir.split('/')[-1] + '_visualize'
        save_folder = os.path.join('/'.join(video_dir.split('/')[:-1]), folder_name)
        os.makedirs(save_folder, exist_ok=True)

    # breakpoint()
    for img_name in ground_truth.keys():
        raw_data = ground_truth[img_name]

        if args.vis_mesh:
            cam_vertices_trans = raw_data['cam_vertices_raw']
            camera_intrinsic = raw_data['camera_intrinsic']

            # print('Start saving mesh')
            save_vis_mesh_folder = os.path.join(save_folder, 'mesh')
            os.makedirs(save_vis_mesh_folder, exist_ok=True)
            save_pcd_img_path = os.path.join(save_vis_mesh_folder, img_name)
            faces = get_faces(helper_folder_path)
            gt_bbox = render_3d_object_on_image(cam_vertices_trans, faces, image_path=os.path.join(video_dir, img_name), output_image_path=save_pcd_img_path, focal_length_x=camera_intrinsic[1][1], focal_length_y=camera_intrinsic[1][1], image_width=camera_intrinsic[0][2]*2, image_height=camera_intrinsic[1][2]*2)
        if args.vis_keypoint:
            # print('Start saving 2d keypoint')
            cam_2djoints = raw_data['gtkps']
            save_2d_dir = os.path.join(save_folder, '2D_keypoint')
            os.makedirs(save_2d_dir, exist_ok=True)
            save_2d_path = os.path.join(save_2d_dir, img_name) 
            visualize_2d(os.path.join(video_dir, img_name), save_2d_path, cam_2djoints[:25])

if __name__=='__main__':
    main()

# python visualize_voccl3d_dataset.py \
# --voccl3d_video_folder /data/AmitRoyChowdhury/Yash/occl_dataset/voccl3d_dataset_release_tiny/scene2_view1/images/SMPLX-female_ulr3_stageii \
# --ground_truth_scene_npy /data/AmitRoyChowdhury/Yash/occl_dataset/voccl3d_dataset_release_tiny/scene2_view1/scene2_view1_ground_truth_labels.npy \
# --helper_folder_path /data/AmitRoyChowdhury/Yash/occl_dataset/voccl3d_dataset_release_helper_files \
# --vis_mesh \
# --vis_keypoint
