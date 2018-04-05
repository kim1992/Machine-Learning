from numpy import *
import matplotlib.pyplot as plt

'''
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - x数据集
        labelMat - y数据集
'''
def loadDataSet(fileName):
    fr = open(fileName)
    numFeat = len(fr.readline().strip().split('\t')) - 1
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = []
        curLine = line.split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):    # 计算回归系数w
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if (linalg.det(xTx) == 0.0): # 奇异矩阵（不可逆）的行列式|A|为0
        print("矩阵为奇异矩阵， 不能求逆")
        return
    ws = xTx.I * xMat.T * yMat  # w = （X.T * X).I * X.T * y
    return ws

xArr, yArr = loadDataSet('ex0.txt')
ws = standRegres(xArr, yArr)

def plotRegression(): # 绘制回归曲线和数据点
    xMat = mat(xArr)    #创建xMat矩阵
    yMat = mat(yArr)    #创建yMat矩阵
    # xCopy = xMat.copy() #深拷贝xMat矩阵
    # xCopy.sort(0)        #排序
    yHat = xMat * ws   #计算对应的y值
    fig = plt.figure()
    ax = fig.add_subplot(111)   #添加subplot
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], c='blue')  # 绘制样本点
    ax.plot(xMat[:, 1], yHat, c = 'red')   #绘制回归曲线
    plt.title("DataSet")
    plt.xlabel('X')
    plt.show()

# plotRegression()

# 计算相关系数
xMat, yMat = mat(xArr), mat(yArr)
yHat = xMat * ws
accuracy = corrcoef(yHat.T, yMat)


'''
    函数说明:使用局部加权线性回归计算回归系数w
    Parameters:
        testPoint - 测试样本点
        xArr - x数据集
        yArr - y数据集
        k - 高斯核的k,自定义参数
    Returns:
        ws - 回归系数
'''
def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * weights * xMat
    if (linalg.det(xTx) == 0.0):
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * xMat.T * weights * yMat
    return testPoint * ws


'''
    函数说明:局部加权线性回归测试
    Parameters:
        testArr - 测试数据集
        xArr - x数据集
        yArr - y数据集
        k - 高斯核的k,自定义参数
    Returns:
        ws - 回归系数
'''
def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def plotlwlrRegression():
    xArr, yArr = loadDataSet('ex0.txt')  # 加载数据集
    yHat_1 = lwlrTest(xArr, xArr, yArr, 1.0)  # 根据局部加权线性回归计算yHat
    yHat_2 = lwlrTest(xArr, xArr, yArr, 0.01)  # 根据局部加权线性回归计算yHat
    yHat_3 = lwlrTest(xArr, xArr, yArr, 0.003)  # 根据局部加权线性回归计算yHat
    xMat = mat(xArr)  # 创建xMat矩阵
    yMat = mat(yArr)  # 创建yMat矩阵
    srtInd = xMat[:, 1].argsort(0)  # 排序，返回索引值
    xSort = xMat[srtInd][:, 0, :]
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], yHat_1[srtInd], c='red')  # 绘制回归曲线
    axs[1].plot(xSort[:, 1], yHat_2[srtInd], c='red')  # 绘制回归曲线
    axs[2].plot(xSort[:, 1], yHat_3[srtInd], c='red')  # 绘制回归曲线
    axs[0].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[1].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    axs[2].scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0].set_title('k=1.0')
    axs1_title_text = axs[1].set_title('k=0.01')
    axs2_title_text = axs[2].set_title('k=0.003')
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()


