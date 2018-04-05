from numpy import *

'''
功能：导入数据集
输入：无
输出：数据矩阵，标签向量
'''
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines(): # 逐行读取，并保存成矩阵，x0为1(bias)
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))    # lineArr[2] 是标签
    return dataMat, labelMat    # 返回数据矩阵和标签

'''
功能：计算x的Sigmoid函数
输入：x
输出：x的Sigmoid函数
'''
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

'''
功能：Logistic回归梯度上升优化算法
输入：无
输出：优化后的权重向量
'''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) # 转换成numpy矩阵数据类型
    labelMat = mat(classLabels).transpose() # 把标签转换成行向量
    m, n = shape(dataMatrix)    # m是行，n是列
    alpha = 0.001   # 向目标移动的步长
    maxCycles = 500 # 迭代次数
    weights = ones((n,1))   # 初始化矩阵向量，n行一列的单位矩阵
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)   # 把z=wx带入函数，求出一个列向量，获得分类标签y=1的概率
        error = labelMat - h     # h是p(y=1)，labelMat是原始样本分类标签y，error其实就是yi-p(y=1)
        weights = weights + alpha * dataMatrix.transpose() * error  #而dataMatrix是原始x，则这里其实就是x*[y-p(y=1)]，获得的是梯度向量，乘以步数alpha就是梯度上升。
    return weights

# 画出决策边界
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()   #导入数据集
    dataMatrix = mat(dataMat)    # 将dataMat转换为array
    n = shape(dataMatrix)[0]   #得dataArr行数
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:   # 标签为1
            xcord1.append(dataMatrix[i,1])
            ycord1.append(dataMatrix[i,2])
        else :                      # 标签为0
            xcord2.append(dataMatrix[i,1])
            ycord2.append(dataMatrix[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's') #red square红方块
    ax.scatter(xcord2, ycord2, s = 30, c = 'green') #绿圆点
    x = arange(-3.0, 3.0, 0.1)  # 在[-3.0,3.0]区间里以0.1的步长取数，得列表
    y = (-weights[0] - weights[1] * x) / weights[2] # 直线方程：weights[0] + weights[1] * x1 + weights[2] * x2 = 0
                                                    # x0 = 1 则 x2 = （-w0 - w1*x1）/ w2
    ax.plot(x,y.transpose())
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

dm,lm = loadDataSet()
# w = gradAscent(dm,lm)
# plotBestFit(w)

'''
功能：随机梯度上升
输入：无
输出：优化后的权重向量
'''
# 区别是这个的h和error是数值，而之前的是向量
# 这个的数据类型是numpy数组，没有矩阵的转换过程
# 可以在新数据到来时就完成参数更新，不需要读取整个数据集进行批处理，占用更少的资源，是在线算法
def stoGradAscent0(dataMatrix, classLabels):
    dataArr = array(dataMatrix)
    m, n = shape(dataArr)   # 得到数据集的行列数量
    alpha = 0.01            # 步长
    weights = ones(n)       # 单位行向量，行数=数据集列数
    for i in range(m):
        h = sigmoid(sum(dataArr[i] * weights))  # 求出这一行向量乘以回归系数，得到一个数值，带入函数中
        error = classLabels[i] - h              # 分别与每行的标签相比较，求误差
        weights = weights + alpha * error * dataArr[i]  # 回归系数等于原来的加上步长乘以误差乘以一行向量。
    return weights

ww = stoGradAscent0(dm, lm)
# print(ww)
# plotBestFit(ww)

# 改进的随机梯度上升算法
def stoGradAscent1(dataMatrix, classLabels, numIter = 150):
    dataArr = array(dataMatrix)
    m, n = shape(dataArr)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01    #这个步长在每次迭代的时候都会调整减小，但因为有常数项，则不会减小到0,保证在多次迭代后新数据仍有影响，学习下模拟退火算法
            randIndex = random.randint(0, len(dataIndex))   #从行数数字中随机算出一行进行计算，随机选取样本来更新回归系数，减少周期波动。
            h = sigmoid(sum(dataArr[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataArr[randIndex]
            del(dataIndex[randIndex])   # 删除第randIndex行，不参与迭代
    return weights

www = stoGradAscent1(dm, lm)
# print(www)
# plotBestFit(www)


# 从疝气病症预测病马的死亡率
'''
功能：预测类别标签
输入：特征向量，回归系数
输出：预测的类别标签
'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

# 用于打开测试集和训练集，并对数据进行格式化处理
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): # 将特征值以浮点型放入lineArr（前21个为特征向量，第22个为标签向量）
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)         # 属性集
        trainingLabels.append(float(currLine[21])) # 标签集
    trainWeights = stoGradAscent1(trainingSet, trainingLabels, 500)
    errorCount = 0.0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(lineArr, trainWeights)) != int(currLine[21]):
            errorCount += 1.0   # 预测标签和验证标签不一致,error + 1
    errorRate = errorCount / numTestVec
    print('The error rate of this test is:', errorRate)
    return errorRate

def multiTest():    # 调用10次colicTest并求结果的平均值
    numTests = 10
    errorSum = 0.0
    for i in range(numTests):
        errorSum += colicTest()
    avgErrorRate = errorSum/float(numTests)
    print('After {0} iterations, the average error rate is:'.format(numTests), avgErrorRate)
