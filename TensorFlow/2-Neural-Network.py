#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 2-Neural-Network.py
# @Author: Jingjie Jin
# @Date  : 2018/2/1


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt

'''
Add_Layer
'''
def add_layer(inputs, in_size, out_size, activation_fuction=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + bias
    if activation_fuction is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_fuction(Wx_plus_b)
    return outputs

'''
导入数据 
'''
# newaxis放在第几个位置，就会在shape里面看到相应的位置增加了一个维数
x_data = np.linspace(-1, 1, 300)[:,np.newaxis] # 300行
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise # x的2次方-0.5

xs = tf.placeholder(tf.float32, [None,1])
ys = tf.placeholder(tf.float32, [None,1])

'''
搭建网络 
'''
l1 = add_layer(xs, 1, 10, activation_fuction=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_fuction=None)

error_sum_of_square = tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1])
loss = tf.reduce_mean(error_sum_of_square)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

'''
可视化
'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data, y_data, s = 10)
plt.ion()   #本次运行请注释，全局运行不要注释
plt.show()


'''
训练 
'''
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})

        lines = ax.plot(x_data, prediction_value, c='r', lw=5)
        plt.pause(0.1)


# sess.close()

