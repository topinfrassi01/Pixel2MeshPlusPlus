# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import tflearn
import numpy as np
import pprint
import pickle
import os

import modules.experiments as experiments
from modules.models_p2mpp import MeshNet
from modules.config import execute
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict


def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    # ---------------------------------------------------------------
    # Set random seed
    print('=> pre-porcessing')
    np.random.seed(cfg.seed)
    tf.set_random_seed(cfg.seed)
    # ---------------------------------------------------------------
    num_blocks = 3
    num_supports = 2
    placeholders = {
        'features': tf.placeholder(tf.float32, shape=(None, 3), name='features'),
        'img_inp': tf.placeholder(tf.float32, shape=(3, 224, 224, 3), name='img_inp'),
        'labels': tf.placeholder(tf.float32, shape=(None, 6), name='labels'),
        'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
        'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
        'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)],  # for laplace term
        'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks - 1)],  # for unpooling
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),
        'sample_coord': tf.placeholder(tf.float32, shape=(43, 3), name='sample_coord'),
        'cameras': tf.placeholder(tf.float32, shape=(3, 5), name='Cameras'),
        'faces_triangle': [tf.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)],
        'sample_adj': [tf.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)],
    }

    step = cfg.p2mpp.test_epoch
    data_list_path = os.path.join(cfg.datalists_base_path, cfg.data_list, "test_list.txt")

    model_dir = os.path.join(cfg.models_path, cfg.p2mpp.name)
    predict_dir = os.path.join(cfg.test_results_path, experiments.create_experiment_name(prefix=[cfg.p2mpp.name, cfg.data_list]))
    mesh_root = os.path.join(cfg.coarse_results_path, cfg.coarse_experiment_name)
    
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
        print('==> make predict_dir {}'.format(predict_dir))
    # -------------------------------------------------------------------
    print('=> build model')
    # Define model
    model = MeshNet(placeholders, logging=True, args=cfg.p2mpp)
    # ---------------------------------------------------------------
    print('=> load data')
    data = DataFetcher(file_list=data_list_path, data_root=cfg.test_models_path,
                       image_root=cfg.images_path, is_val=True, mesh_root=mesh_root)
    data.setDaemon(True)
    data.start()
    # ---------------------------------------------------------------
    print('=> initialize session')
    sesscfg = tf.ConfigProto()
    #pylint: disable=no-member
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    # ---------------------------------------------------------------
    model.load(sess=sess, ckpt_path=model_dir, step=step)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    # Construct Feed dict
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    test_number = data.number
    tflearn.is_training(False, sess)
    print('=> start test stage 2')
    for iters in range(test_number):
        # Fetch training data
        # need [img, label, pose(camera meta data), dataID, mesh]
        img_all_view, labels, poses, data_id, mesh = data.fetch()
        feed_dict.update({placeholders['img_inp']: img_all_view})
        feed_dict.update({placeholders['features']: mesh})
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['cameras']: poses})
        # ---------------------------------------------------------------
        _, out2l = sess.run([model.output1l, model.output2l], feed_dict=feed_dict)
        # ---------------------------------------------------------------
        out3_path = os.path.join(predict_dir, data_id.replace('.dat', '.xyz'))
        np.savetxt(out3_path, out2l)

        print('Iteration {}/{}, Data id {}'.format(iters + 1, test_number, data_id))

    # ---------------------------------------------------------------
    experiments.copy_configurations_to_dir(cfg, predict_dir)
    data.shutdown()
    print('CNN-GCN Optimization Finished!')


if __name__ == '__main__':
    print('=> set config')
    args = execute()
    main(args)
