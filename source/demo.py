# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import os
import tensorflow as tf
from tensorflow.python.util import deprecation
#pylint: disable=protected-access
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tflearn
import numpy as np
import pickle

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import execute
from utils.tools import construct_feed_dict, load_demo_image


def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # ---------------------------------------------------------------
    # Set random seed
    print('=> pre-porcessing')
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
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

    model1_dir = os.path.join(cfg.models_path, 'coarse_mvp2m')
    model2_dir = os.path.join(cfg.models_path, 'refine_p2mpp')
    # -------------------------------------------------------------------
    print('=> build model')
    # Define model
    model1 = MVP2MNet(placeholders, logging=True, args=cfg.mvp2m)
    model2 = P2MPPNet(placeholders, logging=True, args=cfg.p2mpp)
    # ---------------------------------------------------------------
    print('=> load data')
    demo_img_list = ['data/demo/plane1.png',
                     'data/demo/plane2.png',
                     'data/demo/plane3.png']
    img_all_view = load_demo_image(demo_img_list)
    cameras = np.loadtxt('data/demo/cameras.txt')
    # ---------------------------------------------------------------
    print('=> initialize session')
    sesscfg = tf.ConfigProto()
    #pylint: disable=no-member
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    # ---------------------------------------------------------------
    model1.load(sess=sess, ckpt_path=model1_dir, step=50)
    model2.load(sess=sess, ckpt_path=model2_dir, step=10)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    # Construct Feed dict
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    tflearn.is_training(False, sess)
    print('=> start test stage 1')
    feed_dict.update({placeholders['img_inp']: img_all_view})
    feed_dict.update({placeholders['labels']: np.zeros([10, 6])})
    feed_dict.update({placeholders['cameras']: cameras})
    stage1_out3 = sess.run(model1.output3, feed_dict=feed_dict)
    
    print('=> start test stage 2')
    feed_dict.update({placeholders['features']: stage1_out3})
    vert = sess.run(model2.output2l, feed_dict=feed_dict)
    vert = np.hstack((np.full([vert.shape[0],1], 'v'), vert))
    face = np.loadtxt('data/face3.obj', dtype='|S32')
    mesh = np.vstack((vert, face))

    pred_path = 'data/demo/predict.obj'
    np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

    print('=> save to {}'.format(pred_path))

if __name__ == '__main__':
    print('=> set config')
    args=execute()
    main(args)
