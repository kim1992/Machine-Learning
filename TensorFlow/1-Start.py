#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 1-Start.py
# @Author: Jingjie Jin
# @Date  : 2018/1/31



import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt

'''
Session
'''
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],[2]])

# matrix multiply  == np.dot(m1, m2)
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run(product)
    print(result)


'''
Variable
'''
state = tf.Variable(0, name = 'counter')
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


'''
Placeholder
'''
#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: 5., input2: 2.}))


x_data = np.linspace(-1, 1, 300)[:,np.newaxis] # 300行
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise # x的2次方-0.5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data, y_data, s = 10)
plt.show()





