#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : titanic1.py
# @Author: Jingjie Jin
# @Date  : 2018/1/20

import os

# 数据处理
import pandas as pd
import numpy as np
import random
import sklearn.preprocessing as preprocessing

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier,RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#submission_sample = pd.read_csv(path + 'gender_submission.csv')

# print(train.head())
# print(train.describe())

# 观察下各数值变量的协方差
# sns.set(context="paper", font="monospace")
# sns.set(style="white")
# f, ax = plt.subplots(figsize=(10,6))
# train_corr = train.drop('PassengerId',axis=1).corr()
# sns.heatmap(train_corr, ax=ax, vmax=.9, square=True)
# ax.set_xticklabels(train_corr.index, size=15)
# ax.set_yticklabels(train_corr.columns[::-1], size=15)
# ax.set_title('train feature corr', fontsize=20)
# plt.show()

''' 1.(1)Age
由于年龄缺省值我们用[-20]进行填充，然后做年龄的分布以及不同存活情况下的年龄分布:
1. 无论获救与否，Age分布都很宽，小孩和年纪中等偏大的人获救更容易一些；
2. age和survived并不是线性关系,如果用线性模型，这个特征也许需要离散处理,然后作为类别变量代入模型
3. 获救的人之中，年龄缺省更少
'''
# fig, axes = plt.subplots(2,1,figsize=(8,6))
# sns.set_style('white')
# sns.distplot(train.Age.fillna(-20), rug=True, color='b', ax=axes[0])
# ax0 = axes[0]
# ax0.set_title('age distribution')
# ax0.set_xlabel('')
# ax1 = axes[1]
# ax1.set_title('age survived distribution')
# k1 = sns.distplot(train[train.Survived==0].Age.fillna(-20), hist=False, color='r', ax=ax1, label='dead')
# k2 = sns.distplot(train[train.Survived==1].Age.fillna(-20), hist=False, color='g', ax=ax1, label='alive')
# ax1.set_xlabel('')
# ax1.legend(fontsize=16)
# plt.show()

# 男性中老年人多，女性更年轻；小孩中男孩较多
# f, ax = plt.subplots(figsize=(8,3))
# ax.set_title('Sex Age dist', size=20)
# sns.distplot(train[train.Sex=='female'].dropna().Age, hist=False, color='pink', label='female')
# sns.distplot(train[train.Sex=='male'].dropna().Age, hist=False, color='blue', label='male')
# ax.legend(fontsize=15)
# plt.show()

# 仓位等级越高，年龄越偏大，蛮符合常识的
# f, ax = plt.subplots(figsize=(8,3))
# ax.set_title('Pclass Age dist', size=20)
# sns.distplot(train[train.Pclass==1].dropna().Age, hist=False, color='pink', label='P1')
# sns.distplot(train[train.Pclass==2].dropna().Age, hist=False, color='blue', label='p2')
# sns.distplot(train[train.Pclass==3].dropna().Age, hist=False, color='g', label='p3')
# ax.legend(fontsize=15)
# plt.show()

''' 1.(2).Pclass
头等舱（Pclass=1）、商务舱（Pclass=2）、经济舱（Pclass=3）人数对比 :
1、不出所料，经济舱人数遥遥领先 - -。
2、从获救比例来看， 头等舱遥遥领先， 屌丝阵亡比例相当惊人。

观察结果：
1.头等舱获救年龄偏低
2.商务舱小孩照顾的很好
3.屌丝仓同样是小孩获救多
'''
# y_dead = train[train.Survived==0].groupby('Pclass')['Survived'].count()
# y_alive = train[train.Survived==1].groupby('Pclass')['Survived'].count()
# pos = [1, 2, 3]
# ax = plt.figure(figsize=(8,4)).add_subplot(111)
# ax.bar(pos, y_dead, color='r', alpha=0.6, label='dead')
# ax.bar(pos, y_alive, color='g', bottom=y_dead, alpha=0.6, label='alive')
# ax.legend(fontsize=16, loc='best')
# ax.set_xticks(pos)
# ax.set_xticklabels(['Pclass%d'%(i) for i in [1, 2, 3]], size=15)
# ax.set_title('Pclass Surveved count', size=20)
# plt.show()

# pos = range(0,6)
# age_list = []
# for Pclass_ in range(1,4):
#     for Survived_ in range(0,2):
#         age_list.append(train[(train.Pclass == Pclass_)&(train.Survived == Survived_)].Age.values)
# fig, axes = plt.subplots(3,1,figsize=(10,6))
# i_Pclass = 1
# for ax in axes:
#     sns.distplot(age_list[i_Pclass*2-2], hist=False, ax=ax, label='Pclass:%d ,survived:0'%(i_Pclass), color='r')
#     sns.distplot(age_list[i_Pclass*2-1], hist=False, ax=ax, label='Pclass:%d ,survived:1'%(i_Pclass), color='g')
#     i_Pclass += 1
#     ax.set_xlabel('age', size=15)
#     ax.legend(fontsize=15)
# plt.show()

''' 1.(3).Sex
女性生存概率更高，74%，男性仅为18%
'''
# print(train.Sex.value_counts())
# print('************************')
# print(train.groupby('Sex')['Survived'].mean())

'''1.(4).Fare
钱出的多的人，更容易获救
'''

'''1.(5).Sibsp & Parch
亲人数目多少和是否获救不是简单的线性关系
'''

'''1.(6).Embarked
1.Q上岸的很多没有年龄
2.C上岸和S上岸的年龄分布较为相似，区别在于C上岸的年龄分布更加扁平，小孩和老人比例更高
3.从C上岸的人有很高的获救概率
'''

'''2. 简单的特征工程
age, cabin在训练集和待预测集中均有缺失，cabin缺失的个数很多
embarked 在训练集中有2个缺失值
'''
# print(train.isnull().sum())
# print(test.isnull().sum())

# S区登船的人最多，这两个缺失值的乘客来自头等舱，而头等舱中从S登船的人数最多，所以直接用S进行填充。
train.Embarked.fillna('S',inplace=True)

# Cabin为空/非空可以作为一个特征，因此对cabin字段做一下处理
train['Cabin'] = train['Cabin'].isnull().apply(lambda x: 'Null' if x is True else 'Not Null')
test['Cabin'] = test['Cabin'].isnull().apply(lambda x: 'Null' if x is True else 'Not Null')

# Name/Ticket 暂时不考虑
del train['Name'], test['Name']
del train['Ticket'], test['Ticket']

''' 需要将年龄离散化处理，同时还要处理缺省值
    1.为空的归为一类
    2.分类的按年龄分段进行离散
'''
# 以5岁为一个周期离散，同时10以下，60以上的年龄分别归类
def age_map(x):
    if x < 10:
        return '10-'
    if x < 60:
        return '{0}-{1}'.format(x//5*5, x//5*5 + 5)
    elif x >= 60:
        return '60+'
    else:
        return 'Null'

train['Age_map'] = train['Age'].apply(lambda x: age_map(x))
test['Age_map'] = test['Age'].apply(lambda x: age_map(x))
# print(train.groupby('Age_map')['Survived'].agg(['count', 'mean']))

# test有个Fare的缺省
# print(test[test['Fare'].isnull()])
# 取均值
test.loc[test['Fare'].isnull(), 'Fare'] = \
test[(test['Pclass'] == 3) & (test['Embarked'] == 'S') & (test['Sex'] == 'male')].dropna().Fare.mean()
# print(test.iloc[152])

# 数据中Fare分布太宽，做一下scaling，加速模型收敛
scaler = preprocessing.StandardScaler()
fare_scale_param = scaler.fit(train['Fare'].values.reshape(-1, 1))
train['Fare'] = fare_scale_param.transform(train['Fare'].values.reshape(-1, 1))
test['Fare'] = fare_scale_param.transform(test['Fare'].values.reshape(-1, 1))
# 将类别型变量全部onehot
train_x = pd.concat([train[['SibSp','Parch','Fare']], pd.get_dummies(train[['Pclass','Sex','Cabin','Embarked','Age_map']])],axis=1)
train_y = train.Survived
test_x = pd.concat([test[['SibSp','Parch','Fare']], pd.get_dummies(test[['Pclass', 'Sex','Cabin','Embarked', 'Age_map']])],axis=1)

# print(train_x.head())

'''3. Baseline Model
采用Logistic Regression作为baseline model并对参数做简单搜索
'''
base_line_model = LogisticRegression()
param = {'penalty': ['l1', 'l1'],
         'C': [0.1, 0.5, 1.0, 5.0]}
# estimator：所使用的分类器
# param_grid：值为字典或者列表，即需要最优化的参数的取值，
# param_grid =param_test1，param_test1 = {'n_estimators':range(10,71,10)}。
# n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值
# cv :交叉验证参数，默认None，使用5折交叉验证 cv=5
grd = GridSearchCV(estimator=base_line_model, param_grid=param, cv=5, n_jobs=3)
grd.fit(train_x, train_y)
# print(grd.best_estimator_)
# 得到baseline model
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False)


'''4. 将模型训练过程的学习曲线打印出来，观察是否存在overfit/underfit情况
'''
def plot_learning_curve(clf, title, X, y, ylim = None, cv = None, n_jobs = 3, train_sizes = np.linspace(0.05,1.0, 5)):
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    ax = plt.figure().add_subplot(111)
    ax.set_title(title)
    if ylim is not None:
        ax.ylim(*ylim)
    ax.set_xlabel('train num of samples')
    ax.set_ylabel('score')
    ax.plot(train_sizes, train_scores_mean, 'o-', color="b", label='train score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color="r", label='testCV score')

    ax.legend(loc='best')
    plt.show()
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(grd, 'learning_rate', train_x, train_y)
#
# gender_submission = pd.DataFrame({'PassengerId': test.iloc[:,0], 'Survived': grd.predict(test_x)})
# gender_submission.to_csv('submission.csv', index=None)