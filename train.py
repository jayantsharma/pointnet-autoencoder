from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from past.utils import old_div
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import ipdb
import pickle
from scipy.io import savemat
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)  # model
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "data_prep"))
import part_dataset
import show3d_balls
from input_pipeline import input_pipeline
from input_pipeline import ROOT as data_root

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU to use [default: GPU 0]")
parser.add_argument("--model", default="model", help="Model name [default: model]")
parser.add_argument(
    "--category", default=None, help="Which single class to train on [default: None]"
)
parser.add_argument("--log_dir", default="log", help="Log dir [default: log]")
parser.add_argument(
    "--num_point", type=int, default=2048, help="Point Number [default: 2048]"
)
parser.add_argument(
    "--max_epoch", type=int, default=201, help="Epoch to run [default: 201]"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch Size during training [default: 32]",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.001,
    help="Initial learning rate [default: 0.001]",
)
parser.add_argument(
    "--momentum", type=float, default=0.9, help="Initial learning rate [default: 0.9]"
)
parser.add_argument(
    "--optimizer", default="adam", help="adam or momentum [default: adam]"
)
parser.add_argument(
    "--decay_step",
    type=int,
    default=200000,
    help="Decay step for lr decay [default: 200000]",
)
parser.add_argument(
    "--decay_rate",
    type=float,
    default=0.7,
    help="Decay rate for lr decay [default: 0.7]",
)
parser.add_argument(
    "--no_rotation",
    action="store_true",
    help="Disable random rotation during training.",
)
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

sys.path.append("models")
MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model + ".py")
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
os.system("cp %s %s" % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system("cp train.py %s" % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, "log_train.txt"), "w")
LOG_FOUT.write(str(FLAGS) + "\n")

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# # Shapenet official train/test split
# DATA_PATH = os.path.join(BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0')
# TRAIN_DATASET = part_dataset.PartDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, class_choice=FLAGS.category, split='trainval')
# TEST_DATASET = part_dataset.PartDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, class_choice=FLAGS.category, split='test')


def log_string(out_str):
    LOG_FOUT.write(out_str + "\n")
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True,
    )
    learing_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True,
    )
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        pointclouds_pl, labels_pl, fnames = input_pipeline("train", BATCH_SIZE)

        with tf.device("/gpu:" + str(GPU_INDEX)):
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar("bn_decay", bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(
                pointclouds_pl, is_training_pl, bn_decay=bn_decay
            )
            """
            The loss is composed of 2 parts:
            1. Matching distribution loss imposed between predicted and target point clouds via Chamfer or EMD
            2. Self-consistency loss for better alignment of planes imposed on prediction
            """
            matching_loss, end_points = MODEL.get_matching_loss(pred, labels_pl, end_points)
            # end_points["pcloss"] = tf.constant(42)
            consistency_loss = MODEL.get_plane_consistency_loss(pred)
            loss = matching_loss + consistency_loss
            tf.summary.scalar("losses/total", loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar("learning_rate", learning_rate)
            if OPTIMIZER == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)

            grads_and_vars = optimizer.compute_gradients(matching_loss)
            for (grad,var) in grads_and_vars:
                if 'upconv5/weights' in var.name:
                    tf.summary.histogram('gradient/emd/' + var.name, grad)
            grads_and_vars = optimizer.compute_gradients(consistency_loss)
            for (grad,var) in grads_and_vars:
                if 'upconv5/weights' in var.name:
                    tf.summary.histogram('gradient/plane/' + var.name, grad)

            grads_and_vars = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=batch)
            # train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=MAX_EPOCH)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "train"), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, "test"), sess.graph)

        # Init variables
        ckpt_path = tf.train.latest_checkpoint(LOG_DIR)
        if ckpt_path:
            saver.restore(sess, ckpt_path)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        ops = {
            "pointclouds_pl": pointclouds_pl,
            "labels_pl": labels_pl,
            "is_training_pl": is_training_pl,
            "pred": pred,
            "loss": loss,
            "train_op": train_op,
            "merged": merged,
            "step": batch,
            "fnames": fnames,
            "end_points": end_points,
        }

        best_loss = 1e20
        for epoch in range(1, MAX_EPOCH + 1):
            log_string("**** EPOCH %03d ****" % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            # epoch_loss = eval_one_epoch(sess, ops, test_writer)
            # if epoch_loss < best_loss:
            #     best_loss = epoch_loss
            #     save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
            #     log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(
                    sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch
                )
                log_string("Model saved in file: %s" % save_path)


def eval():
    losses = []
    lanewise_losses = {
            "lane1": [],
            "lane2": [],
            "lane3": [],
            "lane4": [],
            "lane5": [],
            "lane6": [],
            "lane7": [],
            "lane8": [],
            "lane9": [],
            "lane10": [],
            "hlane1": [],
            "hlane2": [],
            "hlane3": [],
            "hlane4": [],
            }
    # for ckpt_n in range(50, 361, 50):
    for ckpt_n in [150]:
        with tf.Graph().as_default():
            pointclouds_pl, labels_pl, tf_fname = input_pipeline("test", 1)

            with tf.device("/gpu:" + str(GPU_INDEX)):
                is_training_pl = tf.placeholder(tf.bool, shape=())

                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch = tf.Variable(0)
                bn_decay = get_bn_decay(batch)

                # # Get model and loss
                pred, end_points = MODEL.get_model(
                    pointclouds_pl, is_training_pl, bn_decay=bn_decay
                )
                loss, end_points = MODEL.get_loss(pred, labels_pl, end_points)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)

            # Init variables
            ckpt_path = tf.train.latest_checkpoint(LOG_DIR)
            ckpt_path = "%s/model.ckpt-%d" % (LOG_DIR, ckpt_n)
            print(ckpt_path)
            saver.restore(sess, ckpt_path)
            # init = tf.global_variables_initializer()
            # sess.run(init, {is_training_pl:True})

            total_loss = 0
            num_files = len(os.listdir("{}/test".format(data_root)))  # set manually for now
            # i = 0
            # while True:
            for i in tqdm(range(num_files)):
                # try:
                local, future, predicted, lss, fname = sess.run(
                    [pointclouds_pl, labels_pl, pred, loss, tf_fname],
                    feed_dict={is_training_pl: False},
                )
                local = np.squeeze(local)
                future = np.squeeze(future)
                predicted = np.squeeze(predicted)
                total_loss += lss

                fname = fname[0].decode("utf-8")  # bytearray to string
                fname_splits = fname[:-4].split("_")
                lane, waypt = fname_splits[3:5]
                waypt = int(waypt)
                ref = 1 if len(fname_splits) == 5 else int(fname_splits[5])
                # lanewise_losses[lane][ref-1][waypt-1] = lss
                lanewise_losses[lane].append(lss)

                # Bookkeeping
                i += 1
                pkl_fname = "{}/{}.pkl".format(LOG_DIR, fname[:-4])
                # with open(pkl_fname, "wb") as f:
                data = {
                    "local": local,
                    "gt": future,
                    "predicted": predicted,
                    "loss": lss,
                }
                    # pickle.dump(data, f)
                savemat(pkl_fname[:-4] + ".mat", data)
                # except tf.errors.OutOfRangeError:
                    # print("Iters: {}, Total loss: {:.2f}".format(i, total_loss / i))
                    # losses.append(total_loss / i)
                    # break
            print("Iters: {}, Total loss: {:.2f}".format(i, total_loss / i))
            losses.append(total_loss / i)
    for l in losses:
        print(l)
    with open("lanewise_losses.pkl", "wb") as f:
        lanewise_losses = { k: np.array(v) for k,v in lanewise_losses.items() }
        pickle.dump({ "lanewise_losses": lanewise_losses }, f)
    ipdb.set_trace()


def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps, seg = dataset[idxs[i + start_idx]]
        batch_data[i, ...] = ps
        batch_label[i, :] = seg
    return batch_data, batch_label


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    num_files = len(os.listdir("{}/train".format(data_root)))  # set manually for now
    num_batches = old_div(num_files, BATCH_SIZE)

    loss_sum = 0
    pcloss_sum = 0
    for batch_idx in range(num_batches):
        # # Augment batched point clouds by rotation
        # if FLAGS.no_rotation:
        #     aug_data = batch_data
        # else:
        #     aug_data = part_dataset.rotate_point_cloud(batch_data)
        feed_dict = {ops["is_training_pl"]: is_training}
        summary, step, _, loss_val, pcloss_val, pred_val, fnames = sess.run(
            [
                ops["merged"],
                ops["step"],
                ops["train_op"],
                ops["loss"],
                ops["end_points"]["pcloss"],
                ops["pred"],
                ops["fnames"],
            ],
            feed_dict=feed_dict,
        )
        train_writer.add_summary(summary, step)
        loss_sum += loss_val
        pcloss_sum += pcloss_val

        if (batch_idx + 1) % 10 == 0:
            log_string(" -- %03d / %03d --" % (batch_idx + 1, num_batches))
            log_string("mean loss: %f" % (old_div(loss_sum, 10)))
            log_string("mean pc loss: %f" % (old_div(pcloss_sum, 10)))
            loss_sum = 0
            pcloss_sum = 0


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = old_div(len(TEST_DATASET), BATCH_SIZE)

    log_string(str(datetime.now()))
    log_string("---- EPOCH %03d EVALUATION ----" % (EPOCH_CNT))

    loss_sum = 0
    pcloss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        batch_data, batch_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        feed_dict = {
            ops["pointclouds_pl"]: batch_data,
            ops["labels_pl"]: batch_data,
            ops["is_training_pl"]: is_training,
        }
        summary, step, loss_val, pcloss_val, pred_val = sess.run(
            [
                ops["merged"],
                ops["step"],
                ops["loss"],
                ops["end_points"]["pcloss"],
                ops["pred"],
            ],
            feed_dict=feed_dict,
        )
        test_writer.add_summary(summary, step)
        loss_sum += loss_val
        pcloss_sum += pcloss_val
    log_string("eval mean loss: %f" % (old_div(loss_sum, float(num_batches))))
    log_string("eval mean pc loss: %f" % (old_div(pcloss_sum, float(num_batches))))

    EPOCH_CNT += 1
    return old_div(loss_sum, float(num_batches))


if __name__ == "__main__":
    log_string("pid: %s" % (str(os.getpid())))
    train()
    # eval()
    LOG_FOUT.close()
