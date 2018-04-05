from numpy import *
import matplotlib.pyplot as plt
from Tkinter import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine)) #转化为float类型
        dataMat.append(fltLine)
    return dataMat

def plotDataSet(fileName):
    dataMat = loadDataSet(fileName)
    print(shape(dataMat))
    dataMat = array(dataMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], s = 20, c = 'b')
    plt.title('DataSet')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

'''
函数说明:根据特征切分数据集合
    Parameters:
        dataSet - 数据集合
        feature - 带切分的特征
        value - 该特征的值
    Returns:
        mat0 - 切分的数据集合0
        mat1 - 切分的数据集合1
'''
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

'''
函数说明:生成叶结点
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的均值
'''
def regLeaf(dataSet):
    return mean(dataSet[:, -1])

'''
函数说明:误差估计函数
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的总方差
'''
def regErr(dataSet):
    # 均方差 * 样本个数 = 总方差
    return var(dataSet[:, -1] * shape(dataSet)[0])

'''
函数说明:找到数据的最佳二元切分方式函数
    Parameters:
        dataSet - 数据集合
        leafType - 生成叶结点
        regErr - 误差估计函数
        ops - 用户定义的参数构成的元组
    Returns:
        bestIndex - 最佳切分特征
        bestValue - 最佳特征值
'''
def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    tolS = ops[0]   # tolS允许的误差下降值
    tolN = ops[1]   # tolN切分的最少样本数
    if (len(set(dataSet[:, -1].T.tolist()[0])) == 1):
        # 如果当前所有值相等,则退出。(根据set的特性)
        return None, leafType(dataSet)
    m, n =shape(dataSet)
    S = errType(dataSet)    #默认最后一个特征为最佳切分特征,计算其误差估计
    bestS = float('inf')    #最佳误差
    bestIndex = 0           #最佳特征切分的索引值
    bestValue = 0           #最佳特征值
    for featIndex in range(n - 1):  #遍历所有特征列
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]): #遍历所有特征值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  #根据特征和特征值切分数据集
            if (shape(mat0)[0] < tolN or shape(mat1)[0] < tolN):    #如果数据少于tolN,则退出
                continue
            newS = errType(mat0) + errType(mat1)     #计算误差估计
            if newS < bestS:    #如果误差估计更小,则更新特征索引值和特征值
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if ((S - bestS) < tolS):    #如果误差减少不大则退出
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue) #根据最佳的切分特征和特征值切分数据集合
    if (shape(mat0)[0] < tolN or shape(mat1)[0] < tolN):    #如果切分出的数据集很小则退出
        return None, leafType(dataSet)
    return bestIndex, bestValue #返回最佳切分特征和特征值

'''
函数说明:树构建函数
    Parameters:
        dataSet - 数据集合
        leafType - 建立叶结点的函数
        errType - 误差计算函数
        ops - 包含树构建所有其他参数的元组
    Returns:
        retTree - 构建的回归树
'''
def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    # 选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:    #r如果没有特征,则返回特征值
        return val
    retTree ={} #回归树
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)    #分成左数据集和右数据集
    # 创建左子树和右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

# fileName = 'ex2.txt'
# plotDataSet(fileName)

def isTree(obj):    #判断测试输入变量是否是一棵树
    import types
    return (type(obj).__name__ == 'dict')

def getMean(tree):  # 如果找到两个leaf，则对树进行塌陷处理(即返回树平均值)
    if (isTree(tree['left'])):
        tree['left'] = getMean(tree['left'])
    if (isTree(tree['right'])):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData): # 后剪枝
    if (shape(testData)[0] == 0):   #如果测试集为空,则对树进行塌陷处理
        return getMean(tree)
    if (isTree(tree['left']) or isTree(tree['right'])): #如果有左子树或者右子树,则切分数据集
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if (isTree(tree['left'])):  #处理左子树(剪枝)
        tree['left'] = prune(tree['left'], lSet)
    if (isTree(tree['right'])): #处理右子树(剪枝)
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):  #如果当前结点的左右结点为叶结点
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 计算没有合并的误差
        errorNoMerge = sum((lSet[:, -1] - tree['left']) ** 2) + sum((rSet[:, -1] - tree['right']) ** 2)
        # 计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并的误差
        errorMerge = sum((testData[:, -1] - treeMean) ** 2)
        # 如果合并的误差小于没有合并的误差,则合并
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree




