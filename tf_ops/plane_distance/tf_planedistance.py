from __future__ import division
from __future__ import print_function

from builtins import range
from past.utils import old_div

import random
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.test import compute_gradient, compute_gradient_error

import os, sys
import ipdb
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
plane_distance_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_planedistance_so.so'))

def plane_distance(xyz):
    '''
Computes the distance of nearest neighbors for a pair of point clouds
input: xyz: (batch_size,#points,3)   the point cloud
output: dist: (batch_size,#points)   distance from point to plane fitted to nearest neighbors
output: offset: (batch_size,#points) raw distance
output: normal: (batch_size,#points,3) plane normal
    '''
    return plane_distance_module.plane_distance(xyz)
#@tf.RegisterShape('NnDistance')
#def _nn_distance_shape(op):
    #shape1=op.inputs[0].get_shape().with_rank(3)
    #shape2=op.inputs[1].get_shape().with_rank(3)
    #return [tf.TensorShape([shape1.dims[0],shape1.dims[1]]),tf.TensorShape([shape1.dims[0],shape1.dims[1]]),
        #tf.TensorShape([shape2.dims[0],shape2.dims[1]]),tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
@ops.RegisterGradient('PlaneDistance')
def _plane_distance_grad(op, grad_dist, grad_offset, grad_normal):
    offset=op.outputs[1]
    normal=op.outputs[2]
    return plane_distance_module.plane_distance_grad(offset,normal)

def read_clouds():
    with open("clouds.txt") as f:
        lines = f.readlines()
        vec = [ np.float32(token) for line in lines for token in line.split(',') ]
        vec = np.array(vec)
        vec = np.reshape(vec, [2,2048,3])
    return vec


if __name__=='__main__':
    # random.seed(100)
    # np.random.seed(100)
    with tf.Session() as sess:
        xyz = np.random.randn(32,4096,3).astype('float32')
        # xyz = read_clouds()
        inp = tf.Variable(xyz)
        dist, offset, normal = plane_distance(inp)
        loss = tf.reduce_sum(dist)
        opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.05)
        grad = opt.compute_gradients(loss)#[0][0]
        train = opt.apply_gradients(grad)
        # err = compute_gradient(inp, [2,2048,3], loss, (1,), xyz)
        # err = compute_gradient_error(inp, [2,2048,3], loss, (1,), xyz)
        # print(err)
        # ipdb.set_trace()
        sess.run(tf.initialize_all_variables())
        t0=time.time()
        t1=t0
        best=1e100
        # var = sess.run(inp)
        # print(var[0,0,:])
        # print(var.shape, np.linalg.norm(var))
        # d,o,n,lss,g,_ = sess.run([dist,offset,normal,loss,grad,train])
        # var = sess.run(inp)
        # print(g[0][0][0,0,:])
        # print(var[0,0,:])
        # print(var.shape, np.linalg.norm(var))
        # print(lss)
        # with open("ans_grad_python.txt", "w") as f:
        #     for i in range(2):
        #         for j in range(2048):
        #             print("{:.5f} {:.5f} {:.5f}".format(g[i][j][0], g[i][j][1], g[i][j][2]), file=f)

        for i in range(100):
            trainloss, _ = sess.run([loss, train])
            newt=time.time()
            best=min(best,newt-t1)
            print(i,trainloss,old_div((newt-t0),(i+1)),best)
            t1=newt
        #print sess.run([inp1,retb,inp2,retd])
        #grads=compute_gradient([inp1,inp2],[(16,32,3),(16,32,3)],loss,(1,),[xyz1,xyz2])
        #for i,j in grads:
            #print i.shape,j.shape,np.mean(np.abs(i-j)),np.mean(np.abs(i)),np.mean(np.abs(j))
        #for i in xrange(10):
            #t0=time.time()
            #a,b,c,d=sess.run([reta,retb,retc,retd],feed_dict={inp1:xyz1,inp2:xyz2})
            #print 'time',time.time()-t0
        #print a.shape,b.shape,c.shape,d.shape
        #print a.dtype,b.dtype,c.dtype,d.dtype
        #samples=np.array(random.sample(range(xyz2.shape[1]),100),dtype='int32')
        #dist1=((xyz1[:,samples,None,:]-xyz2[:,None,:,:])**2).sum(axis=-1).min(axis=-1)
        #idx1=((xyz1[:,samples,None,:]-xyz2[:,None,:,:])**2).sum(axis=-1).argmin(axis=-1)
        #print np.abs(dist1-a[:,samples]).max()
        #print np.abs(idx1-b[:,samples]).max()
        #dist2=((xyz2[:,samples,None,:]-xyz1[:,None,:,:])**2).sum(axis=-1).min(axis=-1)
        #idx2=((xyz2[:,samples,None,:]-xyz1[:,None,:,:])**2).sum(axis=-1).argmin(axis=-1)
        #print np.abs(dist2-c[:,samples]).max()
        #print np.abs(idx2-d[:,samples]).max()
