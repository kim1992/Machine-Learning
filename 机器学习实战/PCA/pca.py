from numpy import *
import matplotlib.pyplot as plt
from sklearn import decomposition

def loadDataSet(fileName, delim = '\t'):
    fr = open(fileName)
    dataArr = []
    for line in fr.readlines():
        curLine = line.strip().split(delim)
        fltLine = list(map(float, curLine))
        dataArr.append(fltLine)
    return mat(dataArr)

def pca(dataMat, topNfeat = 9999999):    #dataMat为1000×2的矩阵
    meanVals = mean(dataMat, axis = 0)  # 计算每一列的均值
    meanRemoved = dataMat - meanVals    # 每个向量同时都减去均值

    '''
    若rowvar=0，说明传入的数据一行代表一个样本，若非0，说明传入的数据一列代表一个样本。
    因为newData每一行代表一个样本，所以将rowvar设置为0
    '''
    covMat = cov(meanRemoved, rowvar=0)  # 协方差矩阵：（多维）度量各个维度偏离其均值的程度
    eigVals, eigVects = linalg.eig(mat(covMat)) # eigVals为特征值， eigVects为特征向量

    eigValInd = argsort(eigVals)    # 对特征值，进行从小到大的排序，返回从小到大的index序号

    # -1表示倒序，返回topN的特征值[-1 到 -(topNfeat+1) 但是不包括-(topNfeat+1)本身的倒叙]
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 特征值的逆序就可以得到topNfeat个最大的特征向量

    # 这些特征向量将构成后面对数据进行转换的矩阵，该矩阵则利用N个特征将原始数据转换到新空间中
    redEigVects = eigVects[:, eigValInd]    # 重组 eigVects 最大到最小

    lowDDataMat = meanRemoved * redEigVects #1000×2 * 2×N得到1000×N的低维空间的数据
    # 利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    reconMat = (lowDDataMat * redEigVects.T) + meanVals    #1000×N * N*400 +1000×400重构数据

    return lowDDataMat, reconMat

dm = loadDataSet('testSet.txt')
ld, rm = pca(dm, 1)
# print(shape(ld))

def plotData(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0],
               marker='^', s=50)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0],
               marker='o', s=20, c='r')
    plt.show()

# plotData(dm, rm)

'''
示例：利用PCA对半导体制造数据降维
'''

def replaceNanWithMean():   # 缺失值处理：取平均值代替缺失值，平均值根据非NaN得到
    dataMat = loadDataSet('secom.data',' ')
    numFeat = shape(dataMat)[1]
    for i in range(numFeat):    #遍历数据集每一个维度
        # 遍历数据集每一个维度
        meanVal = mean(dataMat[nonzero(~isnan(dataMat[:, i].A))[0], i])
        # 将该维度中所有NaN特征全部用均值替换
        dataMat[nonzero(isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat

dataMat = replaceNanWithMean()
meanVals = mean(dataMat, 0)
meanRemoved = dataMat - meanVals
covMat = cov(meanRemoved, rowvar = 0)
eigVals, eigVects = linalg.eig(mat(covMat))


pca_sklean = decomposition.PCA()
pca_sklean.fit(replaceNanWithMean())
main_var = pca_sklean.explained_variance_
print(sum(main_var)*0.9)
print(sum(main_var[:6]))
plt.plot(eigVals[:20])
plt.show()
