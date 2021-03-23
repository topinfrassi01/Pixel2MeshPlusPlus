# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import os
import numpy as np
import pickle as pickle
import tensorflow as tf
import glob
from modules.metrics import dice
from modules.config import execute

def main():
    args = execute()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    pred_file_list = os.path.join(args.test_results_path, args.p2mpp_experiment_name, '*.xyz')
    xyz_list_path = glob.glob(pred_file_list)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    #A f1-score will be calculated for each threshold.
    threshold = [0.00005, 0.00010, 0.00015, 0.00020]
    name = {'02828884': 'bench', '03001627': 'chair', '03636649': 'lamp', '03691459': 'speaker', '04090263': 'firearm',
            '04379243': 'table', '04530566': 'watercraft', '02691156': 'plane', '02933112': 'cabinet',
            '02958343': 'car', '03211117': 'monitor', '04256520': 'couch', '04401088': 'cellphone'}
    length = {'02828884': 0, '03001627': 0, '03636649': 0, '03691459': 0, '04090263': 0, '04379243': 0, '04530566': 0,
              '02691156': 0, '02933112': 0, '02958343': 0, '03211117': 0, '04256520': 0, '04401088': 0}
    sum_pred = {'02828884': np.zeros(4), '03001627': np.zeros(4), '03636649': np.zeros(4), '03691459': np.zeros(4),
                '04090263': np.zeros(4), '04379243': np.zeros(4), '04530566': np.zeros(4), '02691156': np.zeros(4),
                '02933112': np.zeros(4), '02958343': np.zeros(4), '03211117': np.zeros(4), '04256520': np.zeros(4),
                '04401088': np.zeros(4)}

    index = 0
    for pred_path in xyz_list_path:
        filename = os.path.basename(pred_path)
        #Ground_thruth contains an image of the reconstructed object in [0] and the point cloud and normals on [1]
        ground_truth_path = os.path.join(args.test_models_path, filename.replace(".xyz", ".dat"))
        ground = pickle.load(open(ground_truth_path, 'rb'), encoding='bytes')[1][:, :3]
        predict = np.loadtxt(pred_path)

        class_id = pred_path.split('/')[-1].split('_')[0]
        length[class_id] += 1.0
        sum_pred[class_id] += dice(predict, ground, threshold, sess)

        index += 1

    log_dir = os.path.join(args.misc_results_path, args.p2mpp_experiment_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, 'f1_score.log')
    
    with open(log_path, 'a+') as log:
        for item in length:
            number = length[item] + 1e-6
            score = sum_pred[item] / number
            log.write(", ".join([item, name[item], str(length[item]), str(score),"\n"]))

    sess.close()


if __name__ == '__main__':
    main()