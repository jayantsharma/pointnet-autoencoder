from glob import glob
import os, sys, pickle, shutil
import ipdb
from tqdm import tqdm

from math import ceil
import numpy as np
from scipy.io import loadmat, savemat
from skimage.io import imread, imshow, imsave
from skimage.util import pad
from skimage.transform import rescale
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
from random import randint


# Intrinsics
fx= 658.77248
fy= 663.25464
px= 636.20736
py= 349.37424
K = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

# SfM reconstruction batch size
TRAJECTORY_BATCH_SIZE = 230
ROOT = '/home/jayant/monkey/grocery_data/Supermarket/data/360extrapolation'


def _parse_example(serialized_record):
    example = tf.parse_single_example(
        serialized_record,
        features={
            "feat": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.string)
        },
        name="features",
    )

    feat = tf.decode_raw(example["feat"], tf.float64)
    label= tf.decode_raw(example["label"], tf.float64)

    feat = tf.cast(tf.reshape(feat, (4096, 6)), tf.float32)
    label = tf.cast(tf.reshape(label, (2048, 6)), tf.float32)

    return feat, label


def preprocess(feat, label):
    # Randomly sample 2048 points from output - works better with upconv layers
    # label = tf.random.shuffle(label)
    # label = label[:2048,:]

    # Each point is XYZRGB, we will predict future pt cloud XYZ from local/input XYZRGB
    label = label[...,:3]

    return feat, label


def input_pipeline(split, batch_size):
    files = tf.data.Dataset.list_files(os.path.join(ROOT, split + '.tfrecord'))
    dataset = tf.data.TFRecordDataset(files)
    if split == 'train':
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(_parse_example, num_parallel_calls=2)
    dataset = dataset.map(preprocess, num_parallel_calls=2)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    feat, label = (
        iterator.get_next()
    )

    # Set shapes - makes life easy
    feat.set_shape([batch_size, 4096, 6])
    label.set_shape([batch_size, 2048, 3])

    return feat, label


def test_pipeline():
    ftrs, lbls = input_pipeline('train', 1)
    with tf.Session() as sess:
        f, l = sess.run([ftrs, lbls])
        print(f.shape, f.dtype)
        print(l.shape, l.dtype)
        savemat('foo.mat', { 'pts_local': np.squeeze(f), 'pts_global': np.squeeze(l) })
        ipdb.set_trace()


def write_point_clouds():
    splits =  [ 'train', 'val', 'test' ]

    def get_tfrecord_example(feat, label):
        example = tf.train.Example(features=tf.train.Features(feature={
            'feat': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feat.tobytes()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
            }))
        return example.SerializeToString()

    for split in splits:
        print(split.upper())
        writer = tf.python_io.TFRecordWriter(os.path.join(ROOT, split + '.tfrecord'))
        data_dir = ROOT
        hash_list = glob('{}/point_cloud*mat'.format(ROOT))
        for s in tqdm(hash_list):
            data = loadmat(s)
            feat, label = data['pts_local'], data['pts_global']
            example = get_tfrecord_example(feat, label)
            writer.write(example)
        writer.close()


if __name__ == '__main__':
    write_point_clouds()
    test_pipeline()
