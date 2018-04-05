#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 5-Overfitting.py
# @Author: Jingjie Jin
# @Date  : 2018/2/2

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

'''
1. Load data
'''
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def add_layer(inputs, in_size, out_size, layer_name, activation_fuction=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + bias
    if activation_fuction is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_fuction(Wx_plus_b)
    tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

# define placeholder for inputs to network

'''dropout 最关键一环！！！ keep_prob'''
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

l1 = add_layer(xs, 64, 100, 'l1', activation_fuction=tf.nn.tanh)
prediction = add_layer(l1, 100, 10, 'l2', activation_fuction=tf.nn.softmax)

'''
2. loss
'''
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

'''
3. train
'''
sess = tf.Session()
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('logs/train', sess.graph)
test_writer = tf.summary.FileWriter('logs/test', sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.6})
    if i % 50 == 0:
        # record loss
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)

