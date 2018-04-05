from math import log # 计算香农熵时候会用到log函数
import operator

# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)   # 计算数据集的输入个数
    labelCounts = {}            # 创建空字典 包含分类结果和对应出现次数

    for featVec in dataSet:     # 对数据集dataSet中的每一行featVec进行循环遍历
        currentLabel = featVec[-1]  # currentLabels为featVec的最后一个元素,即类标签
        # 如果标签currentLabels不在字典对应的key中，则计为0，否则+1
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1

    shannonEnt = 0.0    # 定义香农熵shannonEnt
    for key in labelCounts: # 遍历字典labelCounts中的key，即标签
        prob = float(labelCounts[key]) / numEntries # 计算每一个标签出现的频率，即概率
        shannonEnt -= prob * log(prob,2)            # 根据信息熵公式计算每个标签信息熵并累加到shannonEnt上

    return shannonEnt   # 返回求得的整个标签对应的信息熵

# 创建数据集
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels


# 按照给定特征划分数据集
'''
dataSet:待划分的数据集
axis：划分数据集的特征   代表一个下标
value：特征的返回值
返回符合条件的划分出来的列表
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []          # 定义要返回的数据集
    for featVec in dataSet:     # 遍历数据集中的每个特征，即输入数据
        # 如果列标签对应的值为value，则将该条(行)数据加入到retDataSet中
        if featVec[axis] == value:
            # 取featVec的0-axis个数据，不包括axis，放到reducedFeatVec中
            reducedFeatVec = featVec[:axis]
            # 取featVec的axis+1到最后的数据，放到reducedFeatVec中
            reducedFeatVec.extend(featVec[axis+1:])
            # 将reducedFeatVec添加到分割后的数据集retDataSet中，同时reducedFeatVec，retDataSet中没有了axis列的数据
            retDataSet.append(reducedFeatVec)
    return retDataSet    # 返回分割后的数据集

# 选择使分割后信息增益最大的特征，即对应的列
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1     # 获取特征的数目，从0开始，dataSet[0]是一条数据
    baseEntropy = calcShannonEnt(dataSet) # 计算数据集当前的信息熵
    bestInfoGain = 0.0  # 定义最大的信息增益
    bestFeature = -1    # 定义分割后信息增益最大的特征

    # 遍历特征，即所有的列，计算每一列分割后的信息增益，找出信息增益最大的列
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]    # 取出第i列特征赋给featList
        uniqueVals = set(featList)      # 将特征对应的值放到一个集合中，使得特征列的数据具有唯一性
        newEntropy = 0.0                # 定义分割后的信息熵

        # 遍历特征列的所有值(值是唯一的，重复值已经合并)，分割并计算信息增益
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)     # 按照特征列的每个值进行数据集分割
            prob = len(subDataSet) / float((len(dataSet)))   # 按照特征列的每个值进行数据集分割
            newEntropy += prob * calcShannonEnt(subDataSet)  # 计算分割后的子集的信息熵并相加，得到分割后的整个数据集的信息熵
        infoGain = baseEntropy - newEntropy      # 计算分割后的信息增益
        if (infoGain > bestInfoGain):    # 如果分割后信息增益大于最好的信息增益
            bestInfoGain = infoGain      # 将当前的分割的信息增益赋值为最好信息增益
            bestFeature = i              # 分割的最好特征列赋为i
    return bestFeature       # 返回分割后信息增益最大的特征列

''' 递归构建决策树'''

# 多数表决器，用于处理当用完了所有属性，但是类标签仍然不是唯一的
# 定义标签元字典，key为标签，value为标签的数目
def majority(classList):
    classCount ={}          # 新建一个字典
    for vote in classList:   # 如果标签vote不在字典对应的key中，则计为0，否则+1
        classCount[vote] = classCount.get(vote, 0) + 1
    # 对所有标签按数目排序
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1),revese = True)
    return sortedClassCount[0][0]   # 返回数目最多的标签

# 创建决策树
def createTree(dataSet,labels):
    # 将dataSet的最后一列数据(标签)取出赋给classList，classList存储的是标签列
    classList = [example[-1] for example in dataSet]

    # 判断是否所有的列的标签都一致
    if classList.count(classList[0]) == len(classList):
        return classList[0]     # 直接返回标签列的第一个数据

    # 判断dataSet是否只有一条数据
    if len(dataSet[0]) == 1:    # 返回标签列数据最多的标签
        return majority(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)    # 选择一个使数据集分割后最大的特征列的索引
    bestFeatLabel = labels[bestFeat]                # 找到最好的标签

    myTree = {bestFeatLabel:{}} # 定义决策树，key为bestFeatLabel，value为空
    del(labels[bestFeat])       # 删除labels[bestFeat]对应的元素
    featValues = [example[bestFeat] for example in dataSet] # 取出dataSet中bestFeat列的所有值
    uniqueVals = set(featValues)    # 将特征对应的值放到一个集合中，使得特征列的数据具有唯一性
    for value in uniqueVals:     # 遍历uniqueVals中的值
        subLabels = labels[:]    # 子标签subLabels为labels删除bestFeat标签后剩余的标签
        # myTree为key为bestFeatLabel时的决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree   # 返回决策树


