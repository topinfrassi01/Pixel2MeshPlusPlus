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
from utils.visualize import plot_scatter


def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    # ---------------------------------------------------------------
    # Set random seed
    print('=> pre-processing')
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

    data_list_path = os.path.join(cfg.datalists_base_path, cfg.data_list, "train_list.txt")

    root_dir = os.path.join(cfg.train_results_path, experiments.create_experiment_name(prefix=[cfg.p2mpp.name, cfg.data_list]))

    coarse_mesh_root = os.path.join(cfg.coarse_results_path, cfg.coarse_experiment_name)

    new_model_dir = os.path.join(root_dir, "model")
    log_dir = os.path.join(root_dir, 'logs')
    plt_dir = os.path.join(root_dir, 'plt')
    summaries_dir = os.path.join(root_dir, 'summaries')

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
        print('==> make root dir {}'.format(root_dir))
    if not os.path.exists(new_model_dir):
        os.mkdir(new_model_dir)
        print('==> make model dir {}'.format(new_model_dir))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        print('==> make log dir {}'.format(log_dir))
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)
        print('==> make plt dir {}'.format(plt_dir))
    if not os.path.exists(summaries_dir):
        os.mkdir(summaries_dir)
        print('==> make summaries dir {}'.format(summaries_dir))

    train_loss = open('{}/train_loss_record.txt'.format(log_dir), 'a')
    train_loss.write('Net {} | Start training | lr =  {}\n'.format(cfg.p2mpp.name, cfg.p2mpp.lr))
    # -------------------------------------------------------------------
    print('=> build model')
    # Define model
    model = MeshNet(placeholders, logging=True, args=cfg.p2mpp)
    # ---------------------------------------------------------------
    print('=> load data')
    data = DataFetcher(file_list=data_list_path, data_root=cfg.train_models_path,
                       image_root=cfg.images_path, is_val=False, mesh_root=coarse_mesh_root)
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
    train_writer = tf.summary.FileWriter(summaries_dir, sess.graph, filename_suffix='train')
    # ---------------------------------------------------------------

    if cfg.p2mpp.load_cnn:
        print('=> load pre-trained cnn')
        model.loadcnn(sess=sess, ckpt_path=cfg.p2mpp.pre_trained_cnn_path, step=cfg.p2mpp.cnn_step)
    if cfg.p2mpp.restore:
        print('=> load model')
        model.load(sess=sess, ckpt_path=cfg.p2mpp.restored_model_path, step=cfg.p2mpp.init_epoch)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    # Construct Feed dict
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    train_number = data.number
    step = 0
    tflearn.is_training(True, sess)
    print('=> start train stage 2')

    for epoch in range(cfg.p2mpp.epochs):
        current_epoch = epoch + 1 + cfg.p2mpp.init_epoch
        epoch_plt_dir = os.path.join(plt_dir, str(current_epoch))
        if not os.path.exists(epoch_plt_dir):
            os.mkdir(epoch_plt_dir)
        mean_loss = 0
        all_loss = np.zeros(train_number, dtype='float32')
        for iters in range(train_number):
            step += 1
            # Fetch training data
            # need [img, label, pose(camera meta data), dataID]
            img_all_view, labels, poses, data_id, mesh = data.fetch()
            feed_dict.update({placeholders['features']: mesh})
            feed_dict.update({placeholders['img_inp']: img_all_view})
            feed_dict.update({placeholders['labels']: labels})
            feed_dict.update({placeholders['cameras']: poses})
            # ---------------------------------------------------------------
            _, dists, summaries, _1, out2l = sess.run([model.opt_op, model.loss, model.merged_summary_op, model.output1l, model.output2l], feed_dict=feed_dict)
            # ---------------------------------------------------------------
            all_loss[iters] = dists
            mean_loss = np.mean(all_loss[np.where(all_loss)])
            print('Epoch {}, Iteration {}, Mean loss = {}, iter loss = {}, {}, data id {}'.format(current_epoch, iters + 1, mean_loss, dists, data.queue.qsize(), data_id))
            train_writer.add_summary(summaries, step)
            if (iters + 1) % 1000 == 0:
                plot_scatter(pt=out2l, data_name=data_id, plt_path=epoch_plt_dir)
        # ---------------------------------------------------------------
        # Save model
        model.save(sess=sess, ckpt_path=new_model_dir, step=current_epoch)
        train_loss.write('Epoch {}, loss {}\n'.format(current_epoch, mean_loss))
        train_loss.flush()
    # ---------------------------------------------------------------
    data.shutdown()
    print('CNN-GCN Optimization Finished!')


if __name__ == '__main__':
    print('=> set config')
    args = execute()
    pprint.pprint(vars(args))
    main(args)
