#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 3-TensorBoard.py
# @Author: Jingjie Jin
# @Date  : 2018/2/1

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt

'''
搭建图纸 
'''

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None,1], name='x_input')
    ys = tf.placeholder(tf.float32, [None,1], name='y_input')

def add_layer(inputs, in_size, out_size, n_layer, activation_fuction=None):
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('bias'):
            bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + '/bias', bias)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + bias
        if activation_fuction is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_fuction(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

 ## make up some data
x_data= np.linspace(-1, 1, 300, dtype=np.float32)[:,np.newaxis]
noise=  np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data= np.square(x_data) -0.5 + noise

l1 = add_layer(xs, 1, 10, n_layer=1, activation_fuction=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_fuction=None)

with tf.name_scope('error_sum_of_square'):
    error_sum_of_square = tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1])
with tf.name_scope('loss'):
    loss = tf.reduce_mean(error_sum_of_square)
    tf.summary.scalar('loss', loss)
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(0.1)
with tf.name_scope('train'):
    train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/', sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
