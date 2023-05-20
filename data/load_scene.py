import os

import numpy as np
import json
import cv2


def read_files(basedir, rgb_file, depth_file):
    fname = os.path.join(basedir, rgb_file)
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        convert_fn = cv2.COLOR_BGRA2RGBA
    else:
        convert_fn = cv2.COLOR_BGR2RGB
    img = (cv2.cvtColor(img, convert_fn) / 255.).astype(np.float32) # keep 4 channels (RGBA) if available
    depth_fname = os.path.join(basedir, depth_file)
    depth = cv2.imread(depth_fname, -1).astype(np.uint16)
    return img, depth    

def load_custom_ground_truth_depth(basedir, gt_depth_filenames, image_size, depth_scaling_factor):
    H, W = image_size
    gt_depths = []
    gt_valid_depths = []
    for filename in gt_depth_filenames:
        gt_depth_fname = os.path.join(basedir, filename)
        gt_depth = cv2.imread(gt_depth_fname, -1).astype(np.uint16)
        gt_valid_depth = gt_depth > 0
        gt_depth = (gt_depth / depth_scaling_factor)
        gt_depths.append(np.expand_dims(gt_depth, -1))
        gt_valid_depths.append(gt_valid_depth)
    gt_depths = np.stack(gt_depths, 0)
    gt_valid_depths = np.stack(gt_valid_depths, 0)
    return gt_depths, gt_valid_depths


def load_custom_scene(basedir):
    all_imgs = []
    all_depths = []
    all_valid_depths = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    gt_depth_paths = []

    splits = ['train', 'val', 'test', 'video']

    for s in splits:
        near = 0.62
        far = 2.75
        depth_scaling_factor = 1000.0

        intrinsic = np.loadtxt(os.path.join(basedir, 'intrinsics.txt'))
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        cx = intrinsic[0][2]
        cy = intrinsic[1][2]

        imgs = []
        depths = []
        valid_depths = []
        poses = []
        intrinsics = []

        if os.path.exists(os.path.join(basedir, f'{s}.txt')):
            with open(os.path.join(basedir, f'{s}.txt'), 'r') as fp:
                for line in fp:
                    line = line.strip()
                    rgb_file_path = f'rgb/{line:>06}.png'
                    #depth_file_path = f'depth_d435/{line:>06}.png'
                    depth_file_path = f'_depth_gt/{line:>06}.png'
                    ground_truth_depth_file_path = f'_depth_gt/{line:>06}.png'
                    pose_path = f'_camera_pose/{line:>06}.txt'
                    img, depth = read_files(basedir, rgb_file_path, depth_file_path)
                    if depth.ndim == 2:
                        depth = np.expand_dims(depth, -1)

                    H, W = img.shape[:2]
                    pose = np.loadtxt(os.path.join(basedir, pose_path)) @ np.array([[1, 0, 0, 0], 
                                                                                    [0, -1, 0, 0], 
                                                                                    [0, 0, -1, 0], 
                                                                                    [0, 0, 0, 1]])

                    valid_depth = depth[:, :, 0] > 0 # 0 values are invalid depth
                    depth = (depth / depth_scaling_factor).astype(np.float32)
                    gt_depth_paths.append(ground_truth_depth_file_path)
                    imgs.append(img)
                    depths.append(depth)
                    valid_depths.append(valid_depth)
                    intrinsics.append(np.array((fx, fy, cx, cy)))
                    poses.append(pose)
                counts.append(counts[-1] + len(poses))
                
                if len(imgs) > 0:
                    all_imgs.append(np.array(imgs))
                    all_depths.append(np.array(depths))
                    all_valid_depths.append(np.array(valid_depths))
                all_poses.append(np.array(poses).astype(np.float32))
                all_intrinsics.append(np.array(intrinsics).astype(np.float32))
        else:
            counts.append(counts[-1])
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    imgs = np.concatenate(all_imgs, 0)
    depths = np.concatenate(all_depths, 0)
    valid_depths = np.concatenate(all_valid_depths, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)

    gt_depths, gt_valid_depths = load_custom_ground_truth_depth(basedir, gt_depth_paths, (H, W), depth_scaling_factor)
    
    return imgs, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, gt_depths, gt_valid_depths
