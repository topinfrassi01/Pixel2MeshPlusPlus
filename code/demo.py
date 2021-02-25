# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import pickle
import os
import tensorflow as tf
import tflearn
import numpy as np

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import execute
from utils.tools import construct_feed_dict, load_demo_image
from utils.executable_helpers import *

def main(cfg, demo_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # ---------------------------------------------------------------
    # Set random seed
    print('=> pre-porcessing')
    set_random_seed()

    placeholders = create_placeholders()

    model1_dir = cfg.prod_mvp2m_path
    model2_dir = cfg.prod_p2mpp_path
    # -------------------------------------------------------------------
    print('=> build model')
    # Define model
    model1 = MVP2MNet(placeholders, logging=True, args=cfg)
    model2 = P2MPPNet(placeholders, logging=True, args=cfg)
    # ---------------------------------------------------------------
    print('=> load data')
    with open(os.path.join(cfg.demo_path,"renderings.txt"), "r") as fs:
        content = fs.readlines()

    demo_img_list = [os.path.join(cfg.demo_path,name.strip()) for name in content]
    img_all_view = load_demo_image(demo_img_list)
    cameras = np.loadtxt(os.path.join(cfg.demo_path, '/rendering_metadata.txt'))
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
    pkl = pickle.load(open(os.path.join(cfg.prior_data_path, 'iccv_p2mpp.dat'), 'rb'))
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

    pred_path = os.path.join(cfg.prior_data_path, 'predict.obj')
    np.savetxt(pred_path, mesh, fmt='%s', delimiter=' ')

    print('=> save to {}'.format(pred_path))

if __name__ == '__main__':
    print('=> set config')
    args=execute()
    # TODO : Check the ""
    main(args, "")
