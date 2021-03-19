# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
import tflearn
import numpy as np
import pickle
import os


from modules.models_mvp2m import MeshNetMVP2M
from modules.config import execute
from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict
import modules.experiments as experiments

def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    
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

    train_or_test = "train" if cfg.train else "test"

    step = cfg.mvp2m.test_epoch
    model_dir = os.path.join(cfg.models_path, cfg.mvp2m.name)

    data_list_path = os.path.join(cfg.datalists_base_path, cfg.data_list, "{0}_list.txt".format(train_or_test))
    predict_dir = os.path.join(cfg.coarse_results_path, experiments.create_experiment_name(prefix=[cfg.data_list, train_or_test]))

    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
        print('==> make predict_dir {}'.format(predict_dir))
    # -------------------------------------------------------------------
    print('=> build model')
    # Define model
    
    model = MeshNetMVP2M(placeholders, logging=True, args=cfg.mvp2m)

    # ---------------------------------------------------------------
    print('=> load data')
    data = DataFetcher(file_list=data_list_path,
                       data_root=cfg.train_models_path if cfg.train else cfg.test_models_path,
                       image_root=cfg.images_path, is_val=not cfg.train)
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
    print('=> start test stage 1')
    print("Beginning generation")
    for iters in range(test_number):
        # Fetch training data
        # need [img, label, pose(camera meta data), dataID]
        img_all_view, labels, poses, data_id, _ = data.fetch()

        feed_dict.update({placeholders['img_inp']: img_all_view})
        feed_dict.update({placeholders['labels']: labels})
        feed_dict.update({placeholders['cameras']: poses})
        # ---------------------------------------------------------------
        _1, _2, out3 = sess.run([model.output1, model.output2, model.output3], feed_dict=feed_dict)
        # ---------------------------------------------------------------
        # save 1
        # out1_path = os.path.join(predict_dir, data_id.replace('.dat', '_predict_1.xyz'))
        # np.savetxt(out1_path, out1)
        # # save 2
        # out2_path = os.path.join(predict_dir, data_id.replace('.dat', '_predict_2.xyz'))
        # np.savetxt(out2_path, out2)
        # save 3
        out3_path = os.path.join(predict_dir, data_id.replace('.dat', '_predict.xyz'))
        np.savetxt(out3_path, out3)

        
    experiments.copy_configurations_to_dir(cfg, predict_dir)
    # ---------------------------------------------------------------
    data.shutdown()

    print('CNN-GCN Optimization Finished!')


if __name__ == '__main__':
    print('=> set config')
    args = execute()
    main(args)
