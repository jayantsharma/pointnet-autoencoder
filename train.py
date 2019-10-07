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
from scipy.io import savemat, loadmat
from tqdm import tqdm
from glob import glob
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)  # model
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "data_prep"))
import part_dataset

# import show3d_balls
from input_pipeline import input_pipeline
from input_pipeline import ROOT as data_root

# Graph Conv imports
sys.path.append(os.path.join(ROOT_DIR, "/home/jayant/gae"))
from gae.model import GCNModelAE, GCNModelVAE
from gae.layers import reset_graphconvolution_uid

sys.path.append(os.path.join(ROOT_DIR, "/home/jayant/gcn"))
from gcn.utils import preprocess_features, preprocess_adj

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
    "--batch_size", type=int, default=1, help="Batch Size during training [default: 32]"
)
parser.add_argument(
    "--surface_loss_wt", type=float, default=1, help="Weight for surface loss from GAE"
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight for L2 loss on embedding matrix.",
)
parser.add_argument(
    "--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability)."
)
parser.add_argument(
    "--early_stopping",
    type=int,
    default=10,
    help="Tolerance for early stopping (# of epochs).",
)
parser.add_argument(
    "--momentum", type=float, default=0.9, help="Initial learning rate [default: 0.9]"
)
parser.add_argument(
    "--optimizer", default="adam", help="adam or momentum [default: adam]"
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Initial learning rate [default: 0.001]",
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


def get_consistency_loss_wt(global_step):
    num_files = len(os.listdir("{}/train".format(data_root)))  # set manually for now
    num_batches = old_div(num_files, BATCH_SIZE)
    wait_epochs = 0
    wait_steps = wait_epochs * num_batches
    zero = lambda: 0.0
    one = lambda: 100.0
    return tf.cond(global_step < wait_steps, zero, one)


def construct_feed_dict(features, support, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders["features"]: features})
    feed_dict.update(
        {placeholders["support"][i]: support[i] for i in range(len(support))}
    )
    return feed_dict


def train():
    with tf.Graph().as_default():
        num_partial = 553
        num_pred = 768

        data = loadmat('log_mano_baseline/14090.mat')
        # Center both clouds by the mean of gt cloud (ow they get misaligned because Chamfer opt has no notion of distribution)
        gt = data['gt']
        pred = data['predicted']
        mean = np.mean(gt,0)
        gt -= mean
        pred -= mean
        adj_label = data['adj']
        adj_norm = preprocess_adj(adj_label)

        with tf.device("/gpu:" + str(GPU_INDEX)):
            gt_const = tf.constant(gt)
            real_adj_norm_const = tf.cast(tf.SparseTensor(*adj_norm), tf.float32)
            pred_var = tf.Variable(pred, name='cloud')
            global_step = tf.Variable(0)

            # Construct proxy mesh using NN matching
            matching_loss, pred_gt_matching, gt_pred_matching = MODEL.get_matching_loss(
                tf.expand_dims(pred_var, 0), 
                tf.expand_dims(gt_const, 0)
            )
            emd = MODEL.get_emd(
                tf.expand_dims(pred_var, 0), 
                tf.expand_dims(gt_const, 0)
            )

            chamfer_sum = tf.summary.scalar("losses/chamfer", matching_loss)
            emd_sum = tf.summary.scalar("losses/emd", emd)
            # Get statistics on NN - num_unique_nbrs, median_duplication_count
            _, _, unique_counts = tf.unique_with_counts(
                tf.reshape(pred_gt_matching, [-1])
            )
            num_NN = tf.summary.scalar("NN/num", tf.size(unique_counts))
            median_dup = tf.summary.scalar(
                "NN/median_dup", tf.contrib.distributions.percentile(unique_counts, 50)
            )

            # Define GAE placeholders
            fake_adj_norm_pl = tf.sparse_placeholder(tf.float32)
            dropout = tf.placeholder_with_default(0.0, shape=())

            # GAE
            dims = {"hidden1": 32, "hidden2": 16}
            Dreal = GCNModelAE(
                gt_const,
                real_adj_norm_const,
                3,
                dropout,
                dims=dims,
                name="feature_extractor",
            ).embeddings
            # Mean-center features before running through network
            reset_graphconvolution_uid()
            Dfake = GCNModelAE(
                pred_var,
                fake_adj_norm_pl,
                3,
                dropout,
                dims=dims,
                name="feature_extractor",
                reuse=True,
            ).embeddings

            # Feature reconstruction loss
            pred_gt_matching.set_shape([BATCH_SIZE, num_pred])
            gt_pred_matching.set_shape([BATCH_SIZE, num_pred])
            match_Dfake = tf.gather(Dfake, tf.squeeze(pred_gt_matching, 0))
            match_Dreal = tf.gather(Dreal, tf.squeeze(gt_pred_matching, 0))
            feature_loss = 0.5 * tf.add(
                tf.losses.mean_squared_error(Dreal, match_Dfake),
                tf.losses.mean_squared_error(match_Dreal, Dfake),
            )
            feature_loss_sum = tf.summary.scalar("losses/feature", feature_loss)
            loss = FLAGS.surface_loss_wt * feature_loss

            # Segregate variables
            t_vars = tf.trainable_variables()
            P_vars = [ v for v in tf.trainable_variables() if 'cloud' in v.name ]
            F_vars = [ v for v in tf.trainable_variables() if 'feature_extractor' in v.name ]
            gae_keyed_var_list = {
                "/".join(["gcnmodelae", *v.name[:-2].split("/")[1:]]): v for v in F_vars
            }

            lr = get_learning_rate(global_step)
            opt = tf.train.GradientDescentOptimizer(lr)
            grads_and_vars = opt.compute_gradients(loss, P_vars)
            train_op = opt.apply_gradients(grads_and_vars, global_step)

            # Gradient summaries
            gradient_sum = []
            for (grad,var) in grads_and_vars:
                gradient_sum.append(
                    tf.summary.histogram(var.name, grad)
                )
            gradient_sum = tf.summary.merge(gradient_sum)

            # loss_wt = get_consistency_loss_wt(global_step)
            # tf.summary.scalar("losses/wt", loss_wt)
            # consistency_loss = 1 * MODEL.get_plane_consistency_loss(pred)

            # loss = matching_loss # + consistency_loss
            # tf.summary.scalar("losses/total", loss)

            # grads_and_vars = optimizer.compute_gradients(matching_loss)
            # grads_and_vars = optimizer.compute_gradients(consistency_loss)

            # grads_and_vars = optimizer.compute_gradients(loss)
            # train_op = optimizer.apply_gradients(
            #     grads_and_vars, global_step=global_step
            # )

            # Plot gradients wrt prediction - should be diff when loss includes plane regularization
            # G = tf.get_default_graph()
            # emd_grad = G.get_operation_by_name('gradients/MatchCost_grad/tuple/control_dependency_1').outputs[0]
            # plane_grad = G.get_operation_by_name('gradients_1/PlaneDistance_grad/PlaneDistanceGrad').outputs[0]
            # tf.summary.histogram("emd_prediction_gradient", emd_grad)
            # tf.summary.histogram("plane_prediction_gradient", plane_grad)

        # Add summary writers
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(LOG_DIR)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Write graph
        train_writer.add_graph(sess.graph)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()  # saver = tf.train.Saver(max_to_keep=MAX_EPOCH)
        gae_restorer = tf.train.Saver(var_list=gae_keyed_var_list)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        gae_restorer.restore(
            sess, tf.train.latest_checkpoint("/home/jayant/gae/log1e3")
        )
        start_epoch = 1

        for epoch in range(start_epoch, MAX_EPOCH + 1):
            """
            STEP 1
            Get matching and induce graph
            """
            gt_pc, pred_pc, pgm, gpm, lss = sess.run(
                [gt_const, pred_var, pred_gt_matching, gt_pred_matching, matching_loss],
            )
            # Cut out the batch dim
            pred_pc = np.squeeze(pred_pc)
            pgm = np.squeeze(pgm)
            gpm = np.squeeze(gpm)

            # rev_matching = defaultdict(list)
            # for k, v in enumerate(pgm):
            #     rev_matching[v].append(k)
            fake_adj = np.zeros((num_pred, num_pred))
            for i, row in enumerate(adj_label[pgm, :]):
                nnz = np.nonzero(row)[0]
                # nbrs = [el for n in nnz for el in rev_matching[n]]
                nbrs = [gpm[n] for n in nnz]
                fake_adj[i, nbrs] = 1
            # Set diagonals 0
            for i in range(num_pred):
                fake_adj[i, i] = 0
            fake_adj_norm = preprocess_adj(fake_adj)

            """
            STEP 2
            Compute feature reconstruction loss and train
            """
            feed_dict = {
                fake_adj_norm_pl: fake_adj_norm,
                dropout: FLAGS.dropout,
            }
            _, summary = sess.run([train_op, summary_op], feed_dict=feed_dict)
            train_writer.add_summary(summary, epoch)

            if (epoch % 100 == 0) or (epoch == 1):
                savemat('%s/%d.mat' % (LOG_DIR, epoch), { 
                    'gt': gt_pc, 
                    'pred': pred_pc, 
                    'fake_adj': fake_adj 
                    })
                print("Step: {}, Chamfer loss: {:.4f}".format(epoch, lss))


def get_num_files(split):
    # num_files = len(os.listdir("{}/{}".format(data_root, split)))
    num_files = 5e4
    num_files *= 0.8 if split == "train" else 0.2
    return int(num_files)


def eval():
    with tf.Graph().as_default():
        pointclouds_pl, labels_pl, _, _, _, _, nums = input_pipeline("test")

        with tf.device("/gpu:" + str(GPU_INDEX)):
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)

            # # Get model and loss
            pred, end_points = MODEL.get_model(
                tf.expand_dims(pointclouds_pl, 0), is_training_pl, bn_decay=bn_decay
            )
            loss, end_points, _, _ = MODEL.get_matching_loss(
                pred, tf.expand_dims(labels_pl, 0), end_points
            )
            consistency_loss = MODEL.get_plane_consistency_loss(pred)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Init variables
        ckpt_path = tf.train.latest_checkpoint(LOG_DIR)
        print(ckpt_path)
        saver.restore(sess, ckpt_path)
        # init = tf.global_variables_initializer()
        # sess.run(init, {is_training_pl:True})

        total_consistency_loss = 0
        total_loss = 0
        num_files = get_num_files("train")
        # i = 0
        # while True:
        for i in tqdm(range(1000)):
            # try:
            local, future, predicted, lss, clss, ns = sess.run(
                [pointclouds_pl, labels_pl, pred, loss, consistency_loss, nums],
                feed_dict={is_training_pl: False},
            )
            local = np.squeeze(local)
            future = np.squeeze(future)
            predicted = np.squeeze(predicted)
            n = ns[0][0]
            total_loss += lss
            total_consistency_loss += clss

            # Bookkeeping
            i += 1
            data = {"local": local, "gt": future, "predicted": predicted, "loss": lss}
            # savemat("{}/{}.mat".format(LOG_DIR, n), data)
        print(
            "Iters: {}, Total loss: {:.6f}, Total consistency loss: {:.6f}".format(
                i, total_loss / i, total_consistency_loss / i
            )
        )


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
    num_files = get_num_files("train")  # set manually for now
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
    train()
    # eval()
    LOG_FOUT.close()
