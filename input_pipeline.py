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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import tensorflow as tf
from random import randint, sample, shuffle

# Graph Conv imports
sys.path.append("/home/jayant/gcn")
from gcn.utils import *


# Intrinsics
fx = 658.77248
fy = 663.25464
px = 636.20736
py = 349.37424
K = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

ROOT = "/home/jayant/monkey/MANO/50ksamples"


def _parse_example(serialized_record):
    example = tf.parse_single_example(
        serialized_record,
        features={
            "partial": tf.FixedLenFeature([], tf.string),
            "complete": tf.FixedLenFeature([], tf.string),
            "features": tf.FixedLenFeature([], tf.string),
            "adj_orig_coords": tf.FixedLenFeature([], tf.string),
            "adj_orig_values": tf.FixedLenFeature([], tf.string),
            "adj_label_coords": tf.FixedLenFeature([], tf.string),
            "adj_label_values": tf.FixedLenFeature([], tf.string),
            "adj_norm_coords": tf.FixedLenFeature([], tf.string),
            "adj_norm_values": tf.FixedLenFeature([], tf.string),
            "num": tf.FixedLenFeature([], tf.string),
        },
        name="features",
    )

    # Decode bytes to serialized data type
    partial = tf.decode_raw(example["partial"], tf.float64)
    complete = tf.decode_raw(example["complete"], tf.float64)
    features = tf.decode_raw(example["features"], tf.float64)
    adj_orig_coords = tf.decode_raw(example["adj_orig_coords"], tf.int32)
    adj_orig_values = tf.decode_raw(example["adj_orig_values"], tf.float64)
    adj_label_coords = tf.decode_raw(example["adj_label_coords"], tf.int32)
    adj_label_values = tf.decode_raw(example["adj_label_values"], tf.float64)
    adj_norm_coords = tf.decode_raw(example["adj_norm_coords"], tf.int32)
    adj_norm_values = tf.decode_raw(example["adj_norm_values"], tf.float64)
    num = tf.decode_raw(example["num"], tf.int64)

    # Cast and/or reshape
    partial = tf.cast(tf.reshape(partial, (553, 3)), tf.float32)
    complete = tf.cast(tf.reshape(complete, (768, 3)), tf.float32)
    features = tf.cast(tf.reshape(features, (768, 3)), tf.float32)
    # Wrap in SparseTensor in addition
    adj_orig_coords = tf.cast(tf.reshape(adj_orig_coords, (-1, 2)), tf.int64)
    adj_orig_values = tf.cast(adj_orig_values, tf.float32)
    adj_orig = tf.SparseTensor(adj_orig_coords, adj_orig_values, [768, 768])
    adj_label_coords = tf.cast(tf.reshape(adj_label_coords, (-1, 2)), tf.int64)
    adj_label_values = tf.cast(adj_label_values, tf.float32)
    adj_label = tf.SparseTensor(adj_label_coords, adj_label_values, [768, 768])
    adj_norm_coords = tf.cast(tf.reshape(adj_norm_coords, (-1, 2)), tf.int64)
    adj_norm_values = tf.cast(adj_norm_values, tf.float32)
    adj_norm = tf.SparseTensor(adj_norm_coords, adj_norm_values, [768, 768])

    return partial, complete, features, adj_orig, adj_label, adj_norm, num


def preprocess(feat, label, fname_bytes):
    # Randomly sample 2048 points from output - works better with upconv layers
    # label = tf.random.shuffle(label)
    # label = label[:2048,:]

    # Each point is XYZRGB, we will predict future pt cloud XYZ from local/input XYZRGB
    label = label[:, :3]

    # Randomly sample 2048 points from input and combine with output to autoencode complete space
    feat_sample = tf.random_shuffle(feat)
    feat_sample = feat_sample[:2048, :3]
    label = tf.concat((feat_sample, label), 0)

    return feat, label, fname_bytes


def input_pipeline(split):
    files = tf.data.Dataset.list_files(os.path.join(ROOT, split + ".tfrecord"))
    dataset = tf.data.TFRecordDataset(files)
    if split == "train":
        # pass
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(_parse_example, num_parallel_calls=2)
    # dataset = dataset.map(preprocess, num_parallel_calls=2)
    # dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=100)
    iterator = dataset.make_one_shot_iterator()
    prtl, cmplt, ftrs, adj_orig, adj_label, adj_norm, nums = iterator.get_next()

    # Set shapes - makes life easy
    prtl.set_shape([553, 3])
    cmplt.set_shape([768, 3])
    ftrs.set_shape([768, 3])

    return prtl, cmplt, ftrs, adj_orig, adj_label, adj_norm, nums


def test_pipeline():
    prtl, cmplt, ftrs, adj_orig, adj_label, adj_norm, nums = input_pipeline("train")
    with tf.Session() as sess:
        p, c, f, ao, al, an, n = sess.run([prtl, cmplt, ftrs, adj_orig, adj_label, adj_norm, nums])
        p = np.squeeze(p)
        c = np.squeeze(c)
        f = np.squeeze(f)

        print(n, p.shape, c.shape, f.shape)

        savemat("foo.mat", {"partial": p, "complete": c, "features": f, "num": n})
        ipdb.set_trace()
        print("DONE")


def get_train_test_lists():
    _, _, nums = input_pipeline("train", 1)
    with open("train.list", "w") as f:
        with tf.Session() as sess:
            while True:
                try:
                    n = sess.run(nums)
                    n = n[0][0]
                    print("{:d}".format(n), file=f)
                except tf.errors.OutOfRangeError:
                    break

    _, _, nums = input_pipeline("test", 1)
    with open("test.list", "w") as f:
        with tf.Session() as sess:
            while True:
                try:
                    n = sess.run(nums)
                    n = n[0][0]
                    print("{:d}".format(n), file=f)
                except tf.errors.OutOfRangeError:
                    break


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
    print("Total violating: %d" % foo)


def write_point_clouds():
    splits = ["train", "test"]  # Read splits from files like train.list, test.list

    # Point index (1-based) of middle 2 fingers
    l = 357
    u = 581
    num_pred = 768

    def get_tfrecord_example(
        partial,
        complete,
        features,
        adj_orig_coords,
        adj_orig_values,
        adj_label_coords,
        adj_label_values,
        adj_norm_coords,
        adj_norm_values,
        n,
    ):
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "partial": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[partial.tobytes()])
                    ),
                    "complete": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[complete.tobytes()])
                    ),
                    "features": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[features.tobytes()])
                    ),
                    "adj_orig_coords": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[adj_orig_coords.tobytes()])
                    ),
                    "adj_orig_values": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[adj_orig_values.tobytes()])
                    ),
                    "adj_label_coords": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[adj_label_coords.tobytes()])
                    ),
                    "adj_label_values": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[adj_label_values.tobytes()])
                    ),
                    "adj_norm_coords": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[adj_norm_coords.tobytes()])
                    ),
                    "adj_norm_values": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[adj_norm_values.tobytes()])
                    ),
                    "num": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[np.array([n]).tobytes()])
                    ),
                }
            )
        )
        return example.SerializeToString()

    for split in splits:
        writer = tf.python_io.TFRecordWriter(os.path.join(ROOT, split + ".tfrecord"))
        with open("%s.list" % split) as f:
            idxs = f.read().splitlines()
        for idx in tqdm(idxs):
            s = "{}/{}.pkl".format(ROOT, idx)
            with open(s, "rb") as f:
                data = pickle.load(f)
            points, edges = data["V"], data["E"]

            # Get partial hand and permute order of points
            num_points = points.shape[1]
            wofinger = list(range(l - 1)) + list(range(u, num_points))
            partial = points[:, wofinger].T
            np.random.shuffle(partial)

            # Sample num_pred points and permute order of vertices AND corresponding edges
            sel = sample(range(num_points), num_pred)
            complete = points[:, sel].T
            interim = edges[sel, :]  # Perm in 2 stages, dunno why it doesn't work as edges[sel, sel]
            adj = interim[:, sel]
            # # No permutation for single instance
            # complete = points[:,:num_pred].T
            # adj = edges[:num_pred,:num_pred]

            # Rule out self-connectivity and bring to sparse format
            assert np.linalg.norm(np.diagonal(adj)) == 0
            adj_orig = sp.coo_matrix(adj)
            adj_orig_coords, adj_orig_values, _ = sparse_to_tuple(adj_orig)

            # Self-connections for supervision - inner product decoder will obviously score higher for self-similarity
            adj_label = adj_orig + sp.eye(num_pred)
            adj_label_coords, adj_label_values, _ = sparse_to_tuple(adj_label)

            # Preprocessing for Graph Conv Network
            features = preprocess_features(complete)
            adj_norm_coords, adj_norm_values, _ = preprocess_adj(adj)
            # print(idx, np.linalg.norm(adj_orig_coords), np.linalg.norm(adj_norm_values))

            example = get_tfrecord_example(
                partial,
                complete,
                features,
                adj_orig_coords,
                adj_orig_values,
                adj_label_coords,
                adj_label_values,
                adj_norm_coords,
                adj_norm_values,
                int(idx),
            )
            writer.write(example)
        writer.close()


def distribute_point_clouds():
    vertical_center_lanes = ["lane2", "lane3", "lane4", "lane7", "lane8", "lane9"]
    st_waypts = {ln: randint(1, 20) for ln in vertical_center_lanes}
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
    write_point_clouds()
    test_pipeline()
