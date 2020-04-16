import numpy as np
from rad2sim2rad.experiments.masking_by_moving.evaluation import accumulate_poses, se2, se2_to_xya
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('poses', help='Path to the saved .npz folder')


def to_se3(gt_0_t):
    se3 = np.zeros([len(gt_0_t), 4, 4])
    se3[:] = np.diag(np.ones([4]))
    se3[:, 0:2, 0:2] = gt_0_t[:, 0:2, 0:2]
    se3[:, :-2, -1] = gt_0_t[:, :-1, -1]
    return se3

def save_to_kitti(poses, file):
    gt_0_t, d = accumulate_poses(se2(poses))
    se3 = to_se3(gt_0_t)
    np.savetxt(file, se3[:, :-1].flatten(), newline=' ')
    return se3, gt_0_t

if __name__ == '__main__':
    args = parser.parse_args()
    _, name = os.path.split(args.poses)
    name, _ = os.path.splitext(name)
    x = np.load(args.poses)
    gt = x['gt']
    pred = x['predictions']
     
    base_dir = os.path.join("/Users/robweston/code/devkit/data/", name)
    gt_dir = os.path.join(base_dir, 'gt')
    pred_dir = os.path.join(base_dir, 'pred')

    if not os.path.exists(base_dir):
        os.makedirs(gt_dir)
        os.makedirs(pred_dir)
    
    save_to_kitti(gt, os.path.join(gt_dir, '11.txt'))
    save_to_kitti(pred, os.path.join(pred_dir, '11.txt'))
