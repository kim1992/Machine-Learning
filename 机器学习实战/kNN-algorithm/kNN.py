# -*- coding: utf-8 -*-

from numpy import *
# 导入运算符模块 operator
import operator,os

# 创建数据集, 并返回固定数据集group 和 分类标签labels
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# k-近邻算法

'''
inX :     用于分类的输入向量
dataSet： 训练样本集
labels：  标签向量
k：       选择的最近邻居树木
'''

# 对新数据进行分类
def classify0(inX, dataSet, labels, k):
    # 计算距离
    dataSetSize = dataSet.shape[0] # dataSet.shape[0] 是dataSet的第一维的数目，代表行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet # 扩展输入向量，方便计算每一个数据到该向量的距离
    sqDiffMat = diffMat ** 2 # 数组每个元素进行平方
    sqDistances = sqDiffMat.sum(axis = 1) # 求该数组的行和
    distances = sqDistances ** 0.5 # 为每行元素进行开方 即求出了每个数据和输入数据的距离
    sortedDistIndicies = distances.argsort() #为该距离数组进行升序排序  返回排序结果的下标值
    classCount = {}  #声明一个统计K个数据中类别与相应个数的字典
    for i in range(k): #i从0到k-1进行循环
        voteLabels = labels[sortedDistIndicies[i]] #找到i对应下标的类别
        classCount[voteLabels] = classCount.get(voteLabels, 0) + 1 #该类别如果在字典中存在，则取出其值后加1 如果不存在取默认值0再+1

    # 将字典返回为元祖列表 并依据其第二个元素进行降序排序
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True)

    return sortedClassCount[0][0] #返回频率最高元祖的第一个量即标签

# 读取文本中的数据
def file2matrix(filename):
    fr = open(filename)  # 打开文件
    arrayOfLines = fr.readlines() # 得到一个元素 为文件每一行的列表
    numberOfLines = len(arrayOfLines) # 得到文件的行数
    returnMat = zeros((numberOfLines, 3)) # 创建返回的数据矩阵
    classLabelVector = []    # 创建类标签
    index = 0                # 为了给返回的数据集方便赋值的自增变量

    for line in arrayOfLines:  # 读取文件的每一行并处理
        listFromLine = line.strip().split('\t') # 将每一行去掉换行符并且以\t为分隔符分隔为列表
        returnMat[index,:] = listFromLine[0:3]  # 将列表中的0、1、2号元素赋给returnMat的每一行

        # -1提取列表中的最后一个元素并将其标为int变量 必须明确的通知解释器 否则python语言会将这些元素做字符串处理
        classLabelVector.append(int(listFromLine[-1]))
        index += 1  # 索引加1

    return returnMat, classLabelVector  # 返回数据矩阵和对应的类标签


# 归一化函数
'''
newValue = (oldValue - min) / (max - min)
'''
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 求数据矩阵每一列的最小值
    maxVals = dataSet.max(0) # 求数据矩阵每一列的最大值
    ranges = maxVals - minVals # 求数据矩阵每一列的最大最小值差值
    normDataSet = zeros(shape(dataSet)) # 求与数据矩阵维度相同的初始零矩阵
    m = dataSet.shape[0]     # 返回数据矩阵第一维的数目
    normDataSet = dataSet - tile(minVals, (m, 1))   # 求矩阵每一列减去该列最小值，得出差值 (oldValue - min)
    normDataSet = normDataSet / tile(ranges, (m,1)) # 用求的差值除以最大最小值差值，即数据的变化范围，即归一化 (oldValue - min) / (max - min)
    return normDataSet,ranges, minVals # 返回归一化后的数据，最大最小值差值，最小值

# 分类器测试函数
def datingClassTest():
    hoRatio = 0.10 # 测试集所占的比例
    # 从文件中读取数据
    datingDataMat, datingLabels = file2matrix('/Users/jinjingjie/计算机/机器学习实战/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 对数据进行归一化
    m = normMat.shape[0]               # 求数据的总数目
    numTestVecs = int(m * hoRatio)     # 求测试集的数据数目
    errorCount = 0.0                   # 定义错误数目
    for i in range(numTestVecs):       # 对测试数据进行遍历
        # 对每一条数据进行分类
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # 如果分类结果与实际结果不一致
        if (classifierResult != datingLabels[i]):
            errorCount += 1     # 误分类数加1
            # 输出分类结果和实际的类别
            print('The classifier came back with: {0}, the real answer is: {1}'.format(classifierResult, datingLabels[i]))
    print('错误率是：{0}'.format(errorCount / float(numTestVecs))) # 输出错误率

# 约会网站预测函数
def classifyPerson():
    resultList =['not at all', 'in small doses', 'in large doses'] # 定义分类结果的类别
    # 分别读取输入数据
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year? "))
    iceCream = float(input("liters of ice cream consumd per year? "))
    # 从文件中读取已有数据
    datingDataMat, datingLabels = file2matrix('/Users/jinjingjie/计算机/机器学习实战/machinelearninginaction/Ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 对数据进行归一化
    inArr = array([ffMiles, percentTats, iceCream])     # 将单个输入数据定义成一条数据
    # 对输入数据进行分类
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    # 对输入数据进行分类
    print("You will probably like this person: ", resultList[classifierResult - 1])

#将一个手写字符图像文件转换为一维向量
def img2vector(filename):
    returnVect = zeros((1,1024))    # 定义要返回的向量
    fr = open(filename)             # 打开文件
    for i in range(32):             # 遍历文件中的每一行和每一列
        line = fr.readline()        # 读取一行
        for j in range(32):         # 对读取数据赋值到returnVect中
            returnVect[0,32*i + j] = int(line[j])
    return returnVect               # 返回向量


# 手写数字识别系统的测试函数
def handwritingClassTest():
    hwLabels = []           #定义一个空列表, 记录标签向量
    # 得到存放数据的文件列表
    trainingFileList = os.listdir('/Users/jinjingjie/计算机/机器学习实战/machinelearninginaction/Ch02/digits/trainingDigits')
    m = len(trainingFileList)   # 得到文件的个数 也就是数据集一组数据的个数
    trainingMat = zeros((m,1024)) # 创建空的训练集

    # 创建训练集和标签
    for i in range(m):
        fileName = trainingFileList[i]      # 得到文件名和扩展名
        file = fileName.split('.')[0]       # 得到文件名
        classNum = int(file.split('_')[0])  # 得到该文件的存储的数字
        hwLabels.append(classNum)           # 把类标签放到hwLabels中
        # 把文件变成向量并赋值到trainingMat中
        trainingMat[i,:] = img2vector('/Users/jinjingjie/计算机/机器学习实战/machinelearninginaction/Ch02/digits/trainingDigits/'+fileName)

    # 列出测试目录下的所有文件
    testFileList = os.listdir('/Users/jinjingjie/计算机/机器学习实战/machinelearninginaction/Ch02/digits/testDigits')
    errorCount = 0.0            # 定义错误率
    mTest = len(testFileList)   # 定义测试文件数目
    for i in range(mTest):
        fileName = testFileList[i]          # 得到文件名和扩展名
        file = fileName.split('.')[0]       # 得到文件名
        classNum = int(file.split('_')[0])  # 得到该文件的存储的数字
        # 将测试文件转换成向量
        vectorUnderTest = img2vector('/Users/jinjingjie/计算机/机器学习实战/machinelearninginaction/Ch02/digits/testDigits/'+fileName)
        # 进行分类
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # print('the classifier came back with: {0}, the real answer is: {1}'.format(classifierResult, classNum))
        if (classifierResult != classNum):
            errorCount += 1     # 如果二者不一致，累加错误数量
            # 输出预测类别和实际类别  
            print('the classifier came back with: {0}, the real answer is: {1}'.format(classifierResult, classNum))
    print('the total number of errors is: {0}'.format(errorCount))           # 输出错误个数
    print('the total error rate is: {0}'.format(errorCount/float(mTest)))    # 输出错误个数


handwritingClassTest()