#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 4-Classification.py
# @Author: Jingjie Jin
# @Date  : 2018/2/2

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
搭建网络
'''
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

def add_layer(inputs, in_size, out_size, activation_fuction=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + bias
    if activation_fuction is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_fuction(Wx_plus_b)
    return outputs

# add output layer
prediction = add_layer(xs, 784, 10, activation_fuction=tf.nn.softmax)

'''
Cross entropy loss
'''
# the error between prediction and real_data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

'''
Compute_accuracy
'''
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pred = sess.run(prediction, feed_dict={xs: v_xs})
    # tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
    # 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(v_ys, 1))
    # 把布尔值转换成浮点数，然后取平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result



'''
Training
'''
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

