from numpy import *
import matplotlib.pyplot as plt

'''
    创建单层决策树的数据集
    Parameters:
        无
    Returns:
        dataMat - 数据矩阵
        classLabels - 数据标签
'''
def loadSimpData():
    dataMat = matrix([[1.0, 2.1],
                     [1.5, 1.6],
                     [1.3, 1.0],
                     [1.0, 1.0],
                     [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

dataMat, classLabels = loadSimpData()

'''
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
'''
def showDataSet(dataMat, classLabels):
    data_plus = []
    data_minus = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = array(data_plus)
    data_minus_np = array(data_minus)
    plt.scatter(data_plus_np.transpose()[0], data_plus_np.transpose()[1])
    plt.scatter(data_minus_np.transpose()[0], data_minus_np.transpose()[1])
    plt.show()



'''
    单层决策树分类函数
    Parameters:
        dataMatrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        threshVal - 阈值
        threshIneq - 标志
    Returns:
        retArray - 分类结果
'''
def stumpClassify(dataMat, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMat)[0], 1)) #初始化retArray为1
    '''
    这里lt表示less than，表示分类方式，对于小于阈值的样本点赋值为 -1，
    gt表示greater than，也是表示分类方式，对于大于阈值的样本点赋值为 -1
    '''
    if threshIneq == 'lt':
        retArray[dataMat[:, dimen] <= threshVal] = -1.0 #如果小于阈值,则赋值为-1
    else:
        retArray[dataMat[:, dimen] > threshVal] = -1.0   #如果大于阈值,则赋值为-1
    return retArray

'''
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
'''
def buildStump(dataMat, classLabels, D):
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = mat(zeros((m,1)))
    minError = inf  #最小误差初始化为正无穷大

    for i in range(n):  #遍历所有特征
        rangeMin = dataMat[:,i].min()   #找到特征中最小值
        rangeMax = dataMat[:,i].max()   #找到特征中最大值
        stepSize = (rangeMax - rangeMin)/ numSteps  #计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:    #大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)    #计算阈值
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)   #计算分类结果
                # print("rangeMin:{0}, rangeMax:{1}, stepSize:{2}, j:{3}, predictedVals:{4}"
                #       .format(rangeMin, rangeMax, round(stepSize, 2), j, predictedVals.T))
                errArr = mat(ones((m, 1)))              #初始化误差矩阵
                errArr[predictedVals == labelMat] = 0   #分类正确的,赋值为0
                weightedError = D.transpose() * errArr  #计算误差
                # print("split: dim{0}, thresh: {1}, thresh inequal: {2},".format(i, round(threshVal, 2), inequal),
                #       "the weighted error is {0}".format(weightedError))
                if (weightedError < minError):          #找到误差最小的分类方式
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst

D = mat(ones((5,1)) / 5)
bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
# print("bestStump: {0}, error: {1}, classEst: {2}".format(bestStump, minError, bestClassEst))
'''
经过遍历，我们找到，训练好的最佳单层决策树的最小分类误差为0.2
就是对于该数据集，无论用什么样的单层决策树，分类误差最小就是0.2。这就是我们训练好的弱分类器。
接下来，使用AdaBoost算法提升分类器性能，将分类误差缩短到0.
'''

def adaBoostTrainDS(dataMat, classLabels,numIt = 40):
    weakClassArr = []
    m = shape(dataMat)[0]
    D = mat(ones((m,1)) / m)    #初始化权重
    # print("D original: ", D.T)
    aggClassEst = mat(zeros((m,1)))
    # print("classLabels: ", classLabels)
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataMat, classLabels, D)    #构建单层决策树
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16))) #计算alpha值，其中max(error, 1e-16)保证没有除零溢出
        # print("alpha: ", round(alpha, 2))
        bestStump['alpha'] = alpha      #存储弱学习算法权重
        weakClassArr.append(bestStump)  #存储单层决策树
        # print("classEst: ", classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst) #计算e的指数项
        D = multiply(D, exp(expon))
        D = D / D.sum()         #根据样本权重公式，更新样本权重
        # print("D: ", D.T)

        # 计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)

        '''
        np.sign(a) : 计算各元素的符号值 1（+），0，-1（-）
        aggErrors 判断与真正类别不同的个数（符号不同则标记错误）
        '''
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        # print("aggErros: ", aggErrors.T)
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0.0:    #误差为0时，直接退出循环
            break
    return weakClassArr, aggClassEst


'''
   AdaBoost分类函数
   Parameters:
       datToClass - 待分类样例
       classifierArr - 训练好的分类器
   Returns:
       分类结果
'''
def adaClassify(dataMat, classifierArr):
    m = shape(dataMat)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):     #遍历所有分类器，进行分类
        classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print("aggClassEst: ", aggClassEst.T)
    return sign(aggClassEst)


# wc, agg = adaBoostTrainDS(dataMat, classLabels, 9)
# print("weakClassArr: ", wc)
# print(adaClassify(dataMat, wc).T)
# showDataSet(dataMat, classLabels)

def loadDataSet(fileName):
    fr = open(fileName)
    numFeat = len(fr.readline().split('\t'))
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return mat(dataMat), labelMat


dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
# print(dataArr, labelArr)
weakclassArr, aggclassEst = adaBoostTrainDS(dataArr, labelArr, 50)
testArr, testLabel = loadDataSet('horseColicTest2.txt')
# print(weakclassArr)

trainingPredictions = adaClassify(dataArr, weakclassArr)
print(len(trainingPredictions))
errArr0 = mat(ones((len(dataArr), 1)))
print(len(errArr0))
print("training error rate: {0}%".format(round(errArr0[trainingPredictions != mat(labelArr).T].sum() / float(len(dataArr)), 5) * 100))

testPredictions = adaClassify(testArr, weakclassArr)
print(len(testPredictions))
errArr1 = mat(ones((len(testArr), 1)))
print(len(errArr1))
print("test error rate: {0}%".format(round(errArr1[testPredictions != mat(testLabel).T].sum() / float(len(testArr)), 5) * 100))

'''
    绘制ROC
    Parameters:
    predStrengths - 分类器的预测强度
    classLabels - 类别
    Returns:    无
'''
def plotROC(predStrengths, classLabels):
    # predStrengths: 一个Numpy数组或者一个行向量组成的矩阵，该参数代表的是分类器的预测强度，
    # 在分类器和训练函数将这些数值应用到sign()函数之前，它们就已经产生
    # classLabels: 类别标签
   cur = [1.0, 1.0]    #绘制光标的位置
   ySum = 0.0          #用于计算AUC
   numPosClas = sum(array(classLabels) == 1.0)  # (TP + FN)通过数组过滤方式计算正例的数目，并赋给numPosClas，接着在x轴和y轴的0.0到1.0区间上绘点
   yStep = 1 / float(numPosClas)    # numPosClas = (TP + FN)        #y轴步长
   xStep = 1 / float(len(classLabels) - numPosClas)  # len(classLabels) - numPosClas = (FP + TN)  #x轴步长

   sortedIndicies = predStrengths.argsort()    #预测强度排序

   fig = plt.figure()
   fig.clf()
   ax = plt.subplot(111)
   for index in sortedIndicies.tolist()[0]:
       if (classLabels[index] == 1.0):
           # 每得到一个标签为1.0的类，则要沿着y轴的方向下降一个步长，即降低真阳率
           delX = 0
           delY = yStep
       else:
           # 对于每个其他的标签，则是x轴方向上倒退一个步长（假阴率方向），
           # 代码只关注1这个类别标签，采用1/0标签还是+1/-1标签就无所谓了
           delX = xStep
           delY = 0
           ySum += cur[1]   # 所有高度的和ySum随着x轴的每次移动而渐次增加

       # 一旦决定了在x轴还是y轴方向上进行移动，就可在当前点和新点之间画出一条线段
       ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')  # 绘制ROC
       cur = [cur[0] - delX, cur[1] - delY]
   ax.plot([0,1], [0,1], 'b--')
   plt.title('ROC curve for AdaBoost Horse Colic Detection System')
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   ax.axis([0, 1, 0, 1])
   print('AUC:', ySum * xStep)  # 计算AUC
   plt.show()


plotROC(aggclassEst.T, labelArr)
