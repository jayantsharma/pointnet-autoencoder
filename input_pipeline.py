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
fx = 658.77248
fy = 663.25464
px = 636.20736
py = 349.37424
K = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

# SfM reconstruction batch size
TRAJECTORY_BATCH_SIZE = 230
ROOT = "/home/jayant/monkey/grocery_data/Supermarket/data/small"


def _parse_example(serialized_record):
    example = tf.parse_single_example(
        serialized_record,
        features={
            "feat": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.string),
            "fname": tf.FixedLenFeature([], tf.string),
        },
        name="features",
    )

    feat = tf.decode_raw(example["feat"], tf.float64)
    label = tf.decode_raw(example["label"], tf.float64)
    fname_bytes = example["fname"]

    feat = tf.cast(tf.reshape(feat, (4096, 6)), tf.float32)
    label = tf.cast(tf.reshape(label, (2048, 6)), tf.float32)

    return feat, label, fname_bytes


def preprocess(feat, label, fname_bytes):
    # Randomly sample 2048 points from output - works better with upconv layers
    # label = tf.random.shuffle(label)
    # label = label[:2048,:]

    # Each point is XYZRGB, we will predict future pt cloud XYZ from local/input XYZRGB
    label = label[:, :3]

    # Randomly sample 2048 points from input and combine with output to autoencode complete space
    feat_sample = tf.random_shuffle(feat)
    feat_sample = feat_sample[:2048,:3]
    label = tf.concat((feat_sample, label), 0)

    return feat, label, fname_bytes


def input_pipeline(split, batch_size):
    files = tf.data.Dataset.list_files(os.path.join(ROOT, split + ".tfrecord"))
    dataset = tf.data.TFRecordDataset(files)
    if split == "train":
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(_parse_example, num_parallel_calls=2)
    dataset = dataset.map(preprocess, num_parallel_calls=2)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    feat, label, fname_bytes = iterator.get_next()

    # Set shapes - makes life easy
    feat.set_shape([batch_size, 4096, 6])
    label.set_shape([batch_size, 4096, 3])

    return feat, label, fname_bytes


def test_pipeline():
    ftrs, lbls, fname_bytes = input_pipeline("train", 1)
    with tf.Session() as sess:
        f, l, fb = sess.run([ftrs, lbls, fname_bytes])
        f = np.squeeze(f)
        l = np.squeeze(l)
        fname = fb[0].decode("utf-8")  # bytearray to string

        print(f.shape, np.linalg.norm(f[:, :3], axis=1).max())
        print(
            l.shape,
            np.linalg.norm(l[:, :3], axis=1).min(),
            np.linalg.norm(l[:, :3], axis=1).max(),
        )
        print(fname)

        savemat("foo.mat", {"pts_local": f, "pts_global": l, "fname": fname})
        ipdb.set_trace()
        print('DONE')


def normalize():
    hash_list = glob("{}/point_cloud_*lane*mat".format(ROOT))
    foo = 0
    for s in hash_list:
        data = loadmat(s)
        feat, label = data["sampledPl"], data["sampledPg"]
        norms = np.linalg.norm(feat[:, :3], axis=1)
        unit_ball_radius = max(norms)
        feat[:, :3] = feat[:, :3] / unit_ball_radius
        label[:, :3] = label[:, :3] / unit_ball_radius
        local_max = max(np.linalg.norm(feat[:, :3], axis=1))
        global_min = min(np.linalg.norm(label[:, :3], axis=1))
        if not (local_max <= 1 and global_min > 1):
            print("{} {:0.2f}".format(s.split("/")[-1], global_min))
            foo += 1
        savemat(
            "{}/normalized_{}".format(ROOT, s.split("/")[-1]),
            {"pts_local": feat, "pts_global": label},
        )
    print('Total violating: %d' % foo)


def write_point_clouds():
    splits = ["train"]

    def get_tfrecord_example(feat, label, fname):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "feat": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[feat.tobytes()])
                    ),
                    "label": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[label.tobytes()])
                    ),
                    "fname": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[fname.encode("utf-8")])
                    ),
                }
            )
        )
        return example.SerializeToString()

    for split in splits:
        writer = tf.python_io.TFRecordWriter(os.path.join(ROOT, split + ".tfrecord"))
        data_dir = os.path.join(ROOT, split)
        hash_list = glob("{}/normalized_point_cloud*mat".format(data_dir))
        print("{}: {}".format(split.upper(), len(hash_list)))
        for s in tqdm(hash_list):
            data = loadmat(s)
            feat, label = data["pts_local"], data["pts_global"]
            example = get_tfrecord_example(feat, label, s.split("/")[-1])
            writer.write(example)
        writer.close()


def distribute_point_clouds():
    vertical_center_lanes = ["lane2", "lane3", "lane4", "lane7", "lane8", "lane9"]
    st_waypts = {
        ln: randint(1, 20)
        for ln in vertical_center_lanes
    }
    waypt_range = {ln: range(s, s + 7) for (ln, s) in st_waypts.items()}

    train_path = os.path.join(ROOT, "train")
    test_path = os.path.join(ROOT, "test")
    hash_list = glob("{}/normalized_point_cloud*mat".format(ROOT))

    for s in tqdm(hash_list):
        fname = s.split("/")[-1]
        lane, waypt = fname[:-4].split("_")[3:5]
        waypt = int(waypt)
        if (
            (lane == "lane1" and 16 <= waypt <= 29)
            or (lane == "lane10" and 1 <= waypt <= 15)
            or (lane == "hlane2" and 1 <= waypt <= 20)
            or (lane == "hlane3" and 21 <= waypt <= 40)
        ):
            shutil.move(s, test_path)
        elif lane in vertical_center_lanes and waypt in waypt_range[lane]:
            shutil.move(s, test_path)
        else:
            shutil.move(s, train_path)


if __name__ == "__main__":
    # STEP 1
    # normalize()

    # STEP 2
    # distribute_point_clouds()

    # STEP 3
    write_point_clouds()
    test_pipeline()
