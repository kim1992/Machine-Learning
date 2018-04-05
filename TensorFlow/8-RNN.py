#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 8-RNN.py
# @Author: Jingjie Jin
# @Date  : 2018/2/3

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

# load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.01
training_iters = 100000
batch_size = 128

# MNIST data input(img shape: 28*28)
# time steps
# neurons in hidden layer
# MNIST classes(0-9 digits)
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
#     (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
#     (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
bias = {
#     (128, 0)
    'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
#     (10,)
    'out':tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}

def RNN(X, weights, bias):
    # hidden layer for input to cell
    # X (128 batch, 28 steps, 28 inputs)
    # X==> (128*28, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weights['in']) + bias['in']
    # X_in ==> (128batch * 28steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts(c_state, h_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    '''循环计算过程'''
    # 如果 inputs 为 (batches, steps, inputs) ==> time_major=False;
    # 如果 inputs 为 (steps, batches, inputs) ==> time_major=True;
    outputs, final_states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    # state[0]: c_state ;  state[1]: h_state
    results = tf.matmul(final_states[1], weights['out']) + bias['out']

    # or
    # unpack to list[(batch, outputs)..] * steps
    # outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))
    # results = tf.matmul(outputs[-1], weights['out']) + bias['out']

    return results


pred = RNN(x, weights, bias)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x: batch_xs, y: batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))
        step += 1