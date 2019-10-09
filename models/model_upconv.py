""" TF model for point cloud autoencoder. PointNet encoder, UPCONV decoder.
Using GPU Chamfer's distance loss. Required to have 2048 points.

Author: Charles R. Qi
Date: May 2018
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import math
import sys
import ipdb
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))
import tf_nndistance
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/approxmatch'))
import tf_approxmatch
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/plane_distance'))
import tf_planedistance

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """
    bn = False
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    # assert(num_point==2048)
    point_dim = point_cloud.get_shape()[2].value
    end_points = {}

    input_image = tf.expand_dims(point_cloud, -1)

    # Encoder
    print("Input: {}".format(input_image.shape))
    net = tf_util.conv2d(input_image, 64, [1,point_dim],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=bn, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    print("Global Descriptor: {}".format(global_feat.shape))
    net = tf.squeeze(global_feat, axis=[1,2])
    net = tf_util.fully_connected(net, 1024, bn=bn, is_training=is_training, scope='fc00', bn_decay=bn_decay)

    end_points['embedding'] = net

    # UPCONV Decoder
    net = tf.reshape(net, [-1, 1, 1, 1024])
    net = tf_util.conv2d_transpose(net, 512, kernel_size=[2,2], stride=[1,1], padding='VALID', scope='upconv1', bn=bn, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 256, kernel_size=[3,3], stride=[2,2], padding='VALID', scope='upconv2', bn=bn, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 128, kernel_size=[3,3], stride=[2,2], padding='VALID', scope='upconv3', bn=bn, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 64, kernel_size=[4,5], stride=[2,3], padding='VALID', scope='upconv4', bn=bn, bn_decay=bn_decay, is_training=is_training)
    net = tf_util.conv2d_transpose(net, 3, kernel_size=[1,1], stride=[1,1], padding='VALID', scope='upconv5', activation_fn=None)
    net = net[:,:,:32,:]
    end_points['xyzmap'] = net
    net = tf.reshape(net, [batch_size, -1, 3], name='prediction')
    print("Output: {}".format(net.shape))

    return net, end_points


def get_matching_loss(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3,
    """
    # NN distance
    dists_forward, forward_matching, dists_backward, backward_matching = tf_nndistance.nn_distance(pred, label)
    loss = tf.reduce_mean(dists_forward+dists_backward)
    end_points['pcloss'] = loss
    loss *= 100
    return loss, end_points, forward_matching, backward_matching

    # # EMD
    # match = tf_approxmatch.approx_match(label, pred)
    # matching_loss = tf.reduce_mean(tf_approxmatch.match_cost(label, pred, match))
    # tf.summary.scalar('losses/matching', matching_loss)
    # end_points['pcloss'] = matching_loss
    # return matching_loss, end_points


def get_plane_consistency_loss(pred):
    # Plane Distance
    dist, _, _, _ = tf_planedistance.plane_distance(pred)
    consistency_loss = tf.reduce_mean(dist)
    tf.summary.scalar('losses/consistency', consistency_loss)
    return consistency_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        loss = get_loss(outputs[0], tf.zeros((32,2048,3)), outputs[1])
