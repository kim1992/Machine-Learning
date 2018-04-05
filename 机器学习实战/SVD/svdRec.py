from numpy import *
from numpy import linalg as la

def loadExData():
    return [[1, 1, 1, 4, 4],
            [2, 2, 2, 0, 4],
            [5, 5, 5, 4, 0],
            [1, 1, 0, 2, 2],
            [0, 2, 0, 3, 3],
            [0, 0, 0, 1, 1]]

data = loadExData()
u, sigma, vT = la.svd(data)

# 欧式距离
def euclidSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

# 皮尔逊相关系数
def perasSim(inA, inB):
    # 检查是否有3个或更多的点，如果不存在，则返回1，两向量完全相关。
    if (len(inA) < 3):
        return 1.0
    else:
        return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

# 余弦相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

'''
实例：餐馆菜推荐引擎
需要做到：
    （1）寻找用户没有评级的菜, 即在用户-物品矩阵中的0值;
    （2）在用户没有评级的所有物品中,对每个物品预计一个可能的评级分数.
    （3）对这些物品的评分从高到低进行排序,返回前n个物品
参数：
    dataMat：数据矩阵
    user：用户编号
    simMeas：相似度计算方法
    item：物品编号
Return：

'''

# 计算在给定相似度计算方法的条件下，用户对物品的估计评分值
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: #若某个物品评分值为0，表示用户未对物品评分，则跳过，继续遍历下一个物品
            continue
        # 寻找对物品 j 和item都打过分的用户
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # 计算物品item和j之间的相似度
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        simTotal += similarity
        # ratSimTotal 待推荐物品与用户打过分的物品之间的相似度*用户对物品的打分
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

def recommand(dataMat, user, N = 3, simMeas = cosSim,estMethod = standEst):
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]   # 寻找未评级的物品
    if len(unratedItems) == 0:
        return "you rated everything"
    itemScores = []
    for item in unratedItems:
        # 对每一个未评分物品，调用standEst()来产生该物品的预测得分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 该物品的编号和估计得分值放入一个元素列表itemScores中
        itemScores.append((item, estimatedScore))
        # 对itemScores进行从大到小排序，返回前N个未评分物品
    retItem = sorted(itemScores, key = lambda e:e[1], reverse=True)[:N]
    return retItem

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


# m = mat(loadExData2())
# print(m)
# u, sig, vt = la.svd(m)
# print(sig)
# sig2 = sig ** 2
# print(sum(sig2))
# print(sum(sig2) * 0.9)
#
# print(sum(sig2[:2]))
# print(sum(sig2[:3]))
'''
对给定用户给定物品构建一个评分估计值
用于替换recommend函数重的standEst()
'''
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    # 化为对角阵，用np.diag()函数可破
    sig4 = mat(diag(Sigma[:4]))
    # xformedItems表示物品(item)在4维空间转换后的值(v:右奇异矩阵)
    # V(n*k) = M(m*n).T * U(m*k) * Sigma(k*k).I
    xformedItems = dataMat.T * U[:,:4] * sig4.I

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        # 得到对菜item和j都评过分的用户id,用来计算物品item和j之间的相似度
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        # print("The {0} and {1} similarity is: {2}".format(item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

# mm = mat(loadExData2())
# r = recommand(mm, 1, estMethod=svdEst)

'''
实例：基于SVD的图像压缩

目的：原始图像为 32 * 32 = 1024像素的图像，用SVD对其进行压缩，以节省空间或带宽开销
'''
def printMat(inMat, thresh = 0.5):
    for i in range(32):
        for j in range(32):
            if (float(inMat[i,j]) > thresh):
                print(1, end='')
            else:
                print(0, end='')
        print(" ")

def imgCompress(numSV = 2, thresh = 0.5):
    fr = open('0_5.txt')
    myl = []
    for line in fr.readlines():
       newRow = []
       for i in range(32):
           newRow.append(int(line.strip()[i]))
       myl.append(newRow)
    myMat = mat(myl)
    print("**** original matrix ****")
    # printMat(myMat)
    U, Sigma, Vt = la.svd(myMat)
    sigRecon = diag(Sigma[:numSV])
    reconMat = U[:,:numSV] * sigRecon * Vt[:numSV,:]
    print("**** reconstructed matrix using: {0} singular values".format(numSV))
    printMat(reconMat)
    sumU = shape(U[:, :numSV])[0] * shape(U[:, :numSV])[1]
    sumS = shape(sigRecon)[0]
    sumVt = shape(Vt[:numSV, :])[0] * shape(Vt[:numSV, :])[1]
    print('shape(U[:,:numSV]', sumU)
    print('shape(sigRecon)', sumS)
    print('shape(Vt[:numSV,:])', sumVt)
    print('Sum of pixcel: ',sumU + sumS + sumVt)



imgCompress()





