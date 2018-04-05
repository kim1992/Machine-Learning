from numpy import *
from math import *

def loadDataSet(fileName):
    dataSet = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return dataSet

# 计算两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power((vecA - vecB), 2)))

'''
    随机构建初始质心
    位给定数据集构建一个包含k个随机质心的集合
'''
def randCent(dataSet, k):
    dataMat = mat(dataSet)
    n = shape(dataMat)[1]    #dataSet的列数
    centroids = mat(zeros((k, n)))  # 创建k行n列的矩阵，centroids存放簇中心
    for j in range(n):
        minJ = min(dataMat[:, j])   # 第j列的最小值
        rangeJ = float(max(dataMat[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeans = distEclud, createCent = randCent):
    dataMat = mat(dataSet)
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m, 2))) # 创建一个m行2列的矩阵，第一列存放索引值，第二列存放误差(距离），误差用来评价聚类效果
    centroids = createCent(dataMat, k)  # 随机创建k个质点
    clusterChanged = True   # 标志变量，若为true则继续迭代
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  #遍历每个数据
            # 设置两个变量，分别存放数据点到质心的距离，及数据点属于哪个质心
            minDist = inf
            minIndex = -1
            for j in range(k):  #遍历每个质心
                distJI = distMeans(centroids[j, :], dataMat[i, :]) # 计算距离
                if distJI < minDist:    # 将数据归为最近的质心
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:    # 簇分配结果发生改变，更新标志
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print(centroids)
        for cent in range(k):   # 更新质心
            ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]    # 通过数组过滤来获得给定簇的所有点
            centroids[cent, :] = mean(ptsInClust, axis=0)   # 计算所有点的均值，选项axis=0表示沿矩阵的列方向进行均值计算
    return centroids, clusterAssment

# 二分K-均值算法
def biKmeans(dataSet, k, distMeans = distEclud):
    dataMat = mat(dataSet)
    m = shape(dataMat)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataMat, axis=0).tolist()[0]
    cenList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataMat[j,: ]) ** 2 #计算所有点的误差平方，选项axis=0表示沿矩阵的列方向进行均值计算
    while (len(cenList) < k):   # 当簇小于数目k时
        lowestSSE = inf
        for i in range(len(cenList)):
            # 得到dataMat中行号与clusterAssment中所属的中心编号为i的行号对应的子集数据
            ptsInCurrCluster = dataMat[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # 在给定的簇上进行K-均值聚类,k值为2
            centroidMat, splitClusAss = kMeans(ptsInCurrCluster, 2)
            # 划分后的误差平方和
            sseSplit = sum(splitClusAss[:, 1])
            # 剩余的误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit and notSplit: ", sseSplit, sseNotSplit)
            if ((sseSplit + sseNotSplit) < lowestSSE):
                # 选择使得误差最小的那个簇进行划分
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClusAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 将需要分割的聚类中心下的点进行划分   # 新增的聚类中心编号为len(centList)
        # 将新分出来的两个簇的标号一个沿用它父亲的标号，一个用簇的总数来标号。
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(cenList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print("The bestCentToSplit is: ", bestCentToSplit)
        print("The len of bestClustAss is: ", len(bestClustAss))
        print("bestNewCents  ", bestNewCents)


        cenList[bestCentToSplit] = bestNewCents[0,:]     # 将第一个子簇的质心放在父亲质心的原位置
        cenList.append(bestNewCents[1,:])   # 将第二个子簇的质心添加在末尾
        # print("cl: ", cenList)

        # 由第i个簇分解为j、k两个簇所得到的数据将分解之前的数据替换掉
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],: ] = bestClustAss
    return cenList, clusterAssment

dm = loadDataSet('testSet2.txt')
print(shape(dm))
cl, ca = biKmeans(dm, 3)





