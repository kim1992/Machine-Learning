#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Movie-RS.py
# @Author: Jingjie Jin
# @Date  : 2018/2/6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset

# from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import io
import sys
from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader

# 1. 训练推荐模型 步骤:1
def getSimModel():
    '''默认载入movielens数据集'''
    file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_from_file(file_path, reader=reader)
    # data = Dataset.load_builtin('ml-1m')
    trainset = data.build_full_trainset()
    # 使用pearson_baseline方式计算相似度  False以item为基准计算相似度 本例为电影之间的相似度
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    # 使用KNNBaseline算法
    algo = KNNBaseline(sim_options=sim_options)
    # 训练模型
    algo.fit(trainset)
    return algo

# 2. 获取id到name的互相映射
def read_item_names():
    '''获取电影名到电影id 和 电影id到电影名的映射'''
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]
    return rid_to_name, name_to_rid

# 3. 基于之前训练的模型 进行相关电影的推荐
def showSimilarMovies(algo, movieName, rid_to_name, name_to_rid):
    # 获得电影的raw_id
    movie_raw_id = name_to_rid[movieName]
    # 把电影的raw_id转换为模型的内部id
    movie_inner_id = algo.trainset.to_inner_iid(movie_raw_id)
    # 通过模型获取推荐电影(10部)
    movie_neighbors = algo.get_neighbors(movie_inner_id, 10)
    # 模型内部id转换为实际电影id
    neighbors_raw_ids = [algo.trainset.to_raw_iid(inner_id)
                         for inner_id in movie_neighbors]
    # 通过电影id列表 或得电影推荐列表
    neighbors_movies = [rid_to_name[raw_id] for raw_id in neighbors_raw_ids]
    print()
    print('The 10 nearest neighbors of {0} are:'.format(movieName))
    for movie in neighbors_movies:
        print(movie)

 # 获取id到name的互相映射
rid_to_name, name_to_rid = read_item_names()

# 训练推荐模型
algo = getSimModel()

# 显示相关电影
showSimilarMovies(algo, 'Titanic (1997)', rid_to_name, name_to_rid)

