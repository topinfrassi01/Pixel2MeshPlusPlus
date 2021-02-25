# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import pprint
import pickle
import os

import tensorflow as tf
import tflearn
import numpy as np

from modules.models_p2mpp import MeshNet
from modules.config import execute
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict
from utils.executable_helpers import *


def main(cfg, model_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    # ---------------------------------------------------------------
    # Set random seed
    print('=> pre-processing')
    set_random_seed()

    placeholders = create_placeholders()

    step = cfg.test_epoch
    predict_dir = cfg.predictions_p2mpp_path

    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
        print('==> make predict_dir {}'.format(predict_dir))
    # -------------------------------------------------------------------
    print('=> build model')
    # Define model
    model = MeshNet(placeholders, logging=True, args=cfg)
    # ---------------------------------------------------------------
    print('=> load data')
    data = DataFetcher(file_list=cfg.test_file_path, data_root=cfg.test_data_path,
                       image_root=cfg.test_image_path, is_val=True, mesh_root=cfg.test_mesh_root)
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
    pkl = pickle.load(open(os.path.join(cfg.prior_data_path, 'iccv_p2mpp.dat'), 'rb'))
    # Construct Feed dict
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    test_number = data.number
    tflearn.is_training(False, sess)
    print('=> start test stage 2')
    for iters in range(test_number):
        # Fetch training data
        # need [img, label, pose(camera meta data), dataID]
        img_all_view, labels, poses, data_id, mesh = data.fetch()
        feed_dict.update({placeholders['img_inp']: img_all_view})
        feed_dict.update({placeholders['features']: mesh})
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['cameras']: poses})
        # ---------------------------------------------------------------
        _1, out2l = sess.run([model.output1l, model.output2l], feed_dict=feed_dict)
        # ---------------------------------------------------------------
        # save GT
        label_path = os.path.join(predict_dir, data_id.replace('.dat', '_ground.xyz'))
        np.savetxt(label_path, labels)
        # save 1
        # out1_path = os.path.join(predict_dir, data_id.replace('.dat', '_predict_1.xyz'))
        # np.savetxt(out1_path, out1)
        # # save 2
        # out2_path = os.path.join(predict_dir, data_id.replace('.dat', '_predict_2.xyz'))
        # np.savetxt(out2_path, out2)
        # save 3
        out3_path = os.path.join(predict_dir, data_id.replace('.dat', '_predict.xyz'))
        np.savetxt(out3_path, out2l)

        print('Iteration {}/{}, Data id {}'.format(iters + 1, test_number, data_id))

    data.shutdown()
    print('CNN-GCN Optimization Finished!')


if __name__ == '__main__':
    print('=> set config')
    args = execute()
    pprint.pprint(vars(args))
    
    # TODO : Check the ""
    main(args,"")
