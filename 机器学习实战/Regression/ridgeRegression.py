from numpy import *
import matplotlib.pyplot as plt
import regression as rg
import abalone as ab

'''
    函数说明:岭回归
    Parameters:
        xMat - x数据集
        yMat - y数据集
        lam - 缩减系数
    Returns:
        ws - 回归系数
'''
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam # 分母
    if (linalg.det(denom) == 0.0):
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * xMat.T * yMat
    return ws

'''
    函数说明:岭回归测试
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        wMat - 回归系数矩阵
'''
def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    # 数据标准化
    yMean = mean(yMat, axis = 0)    #行与行操作，求均值
    yMat = yMat - yMean             #行与行操作，求均值
    xMeans = mean(xMat, axis = 0)   #行与行操作，求均值
    xVar = var(xMat, axis = 0)      #行与行操作，求方差
    xMat = (xMat - xMeans) / xVar   #数据减去均值除以方差实现标准化
    numTestPts = 30                 #30个不同的lambda测试
    wMat = zeros((numTestPts, shape(xMat)[1]))  #初始回归系数矩阵
    for i in range(numTestPts):     #改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, exp(i - 10))   #lambda以e的指数变化，最初是一个非常小的数
        wMat[i, :] = ws.T           #计算回归系数矩阵
    return wMat


'''
绘制岭回归系数矩阵
'''
def plotwMat():
    abX, abY = rg.loadDataSet('abalone.txt')
    redgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(redgeWeights)
    ax_title_text = ax.set_title('log(lambada) with w')
    ax_xlabel_text = ax.set_xlabel('log(lambada)')
    ax_ylabel_text = ax.set_ylabel('w')
    plt.show()

'''
函数说明:数据标准化
    Parameters:
        xMat - x数据集
        yMat - y数据集
    Returns:
        inxMat - 标准化后的x数据集
        inyMat - 标准化后的y数据集
'''
def regularize(xMat, yMat):
    inxMat = xMat.copy()
    inyMat = yMat.copy()
    yMean = mean(yMat, 0)
    inyMat = yMat - yMean
    inxMeans = mean(inxMat, 0)
    inVar = var(inxMat, 0)
    inxMat = (inxMat - inxMeans) / inVar
    return inxMat, inyMat

'''
    函数说明:前向逐步线性回归
    Parameters:
        xArr - x输入数据
        yArr - y预测数据
        eps - 每次迭代需要调整的步长
        numIt - 迭代次数
    Returns:
        returnMat - numIt次迭代的回归系数矩阵
'''
def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xMat, yMat = regularize(xMat, yMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    # 迭代numIt次，每次迭代，循环n*2次(每个特征有增大和减小两个方向)，
    # 找出令rssError最小的方向(哪个特征，对应增大还是减小),保存ws,下次迭代在ws基础上做更新
    for i in range(numIt):   #迭代numIt次
        lowestError = float('inf')
        for j in range(n):  #遍历每个特征的回归系数
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign #微调回归系数
                yTest = xMat * wsTest   #计算预测值
                rssE = ab.rssError(yMat.A, yTest.A)  #计算平方误差
                if rssE < lowestError:   #如果误差更小，则更新当前的最佳回归系数
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T  #记录numIt次迭代的回归系数矩阵
    return returnMat

'''
绘制岭回归系数矩阵
'''
def plotstageWiseMat():
    xArr, yArr = rg.loadDataSet('abalone.txt')
    returnMat = stageWise(xArr, yArr, 0.005, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(returnMat)
    ax_title_text = ax.set_title('numIt with w')
    ax_xlabel_text = ax.set_xlabel('numIt')
    ax_ylabel_text = ax.set_ylabel('w')
    plt.show()

plotstageWiseMat()