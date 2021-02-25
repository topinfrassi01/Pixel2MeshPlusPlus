import os
from datetime import datetime

import numpy as np
import tensorflow as tf


def create_placeholders():
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

    return placeholders


def set_random_seed():
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)


def create_train_folder_architecture(save_path):
    tag_name = datetime.now().strftime("%Y%m%d-%H:%M")
    root_dir = os.path.join(save_path, tag_name)

    model_dir = os.path.join(root_dir, 'models')
    log_dir = os.path.join(root_dir, 'logs')
    plt_dir = os.path.join(root_dir, 'plt')

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(plt_dir):
        os.mkdir(plt_dir)

    summaries_dir = os.path.join(root_dir, 'summaries')

    return model_dir, log_dir, plt_dir, summaries_dir