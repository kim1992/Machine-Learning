from numpy import *

'''
功能：导入数据集
输入：文件名
输出：数据矩阵，标签向量
'''
def loadDataSet(fileName):
    dataMat = []    # 数据矩阵
    labelMat = []   # 标签向量
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append([float(lineArr[2])])
    return dataMat, labelMat

'''
功能：在(0, m)的区间范围内随机选择一个除i以外的整数
输入：不能选择的整数i，区间上界m
输出：随机选择的整数
'''
def selectJrand(i, m):
    j = i
    while(j == i): # 不用if的原因是防止随机选的整数又等于i
        j = random.randint(0, m)
    return

'''
功能：保证aj在区间[L, H]里面
输入：要调整的数aj，区间上界H，区间下界L
输出：调整好的数aj
'''
def clipAlpha(aj, H, L):
    if(aj > H):
        aj = H
    if (aj < L):
        aj = L
    return aj

'''
功能：简化版SMO算法
输入：数据矩阵dataMatIn，标签向量classLabels，常数C，容错率toler，最大迭代次数maxIter
输出：超平面位移项b，拉格朗日乘子alpha
'''
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMat = mat(dataMatIn)
    labelMat = classLabels
    b = 0
    m, n = shape(dataMat)   #数据矩阵行数和列数，表示训练样本个数和特征值个数
    alphas = mat(zeros((m, 1))) #m*1阶矩阵
    iter = 0
    while (iter < maxIter):     #循环直到超出最大迭代次数
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T *
                        (dataMat * dataMat[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            #误差很大，可以对该数据实例所对应的alpha值进行优化
            if (((labelMat[i] * Ei < - toler) and (alphas[i] < C)) or
                ((labelMat[i] * Ei > toler) and (alphas[i] > 0))):
                j = selectJrand(i, m)   #在(0, m)的区间范围内随机选择一个除i以外的整数，即随机选择第二个alpha
                # 求变量alphaJ对应的误差
                fXj = float(multiply(alphas, labelMat).T * (dataMat * dataMat[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # 不能直接 alphaIold = alphas[i]，否则alphas[i]和alphaIold指向的都是同一内存空间
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if (L == H):
                    print("L == H")
                    continue
                eta = 2.0 * dataMat[i, :] * dataMat[j, :].T - \
                      dataMat[i, :] * dataMat[i, :].T - \
                      dataMat[j, :] * dataMat[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)  # 保证alphas[j]在区间[L, H]里面
                # 检查alpha[j]是否有较大改变，如果没有则退出for循环
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough")
                    continue
                # labelMat[i]与labelMat[j]绝对值均为1，则alphas[i]与alphas[j]改变大小一样
                # 保证alpha[i] * labelMal[i] + alpha[j] * labelMal[j] = c
                # 即Delta(alpha[i]) * labelMal[i] + Delta(alpha[j]) * labelMal[j] = 0
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 对alpha和alpha[j]进行优化之后，给这两个alpha值设置一个常数项b
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMat[i, :] * dataMat[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMat[i, :] * dataMat[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                              dataMat[i, :] * dataMat[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMat[j, :] * dataMat[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
                # 不是1，这个迭代思路比较巧妙，是以最后一次迭代没有误差为迭代结束条件
            if (alphaPairsChanged == 0):
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)
        return b, alphas

'''
功能：建立数据结构用于保存所有的重要值
输入：无
输出：无
'''
class optStruct:
    def __init__(self, dataMatIn, classLabes, C, toler, kTup):#__init__作用是初始化已实例化后的对象
        self.X = dataMatIn
        self.labelMat = classLabes
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]#dataMatIn行数
        self.alphas = mat(zeros((self.m, 1)))#(self.m, 1)是一个元组，下同
        self.b = 0
        # m*2误差矩阵，第一列为eCache是否有效的标志位，第二列是
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))#建立核矩阵
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

'''
功能：计算第k个alpha的误差值
输入：数据集，alpha数
输出：误差值
'''
def calcEk(oS, k):
    #fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk -float(oS.labelMat[k])
    return Ek

'''
功能：选择有最大步长的alpha值
输入：第一个alpha值，数据集，第一个alpha对应的误差值
输出：第二个alpha值和对应的误差值
'''
def selectJ(i, oS, Ei):
    maxK = -1#最大步长对应j值
    maxDeltaE = 0#最大步长
    Ej = 0#最大误差值
    oS.eCache[i] = [1, Ei]#使i值对应的标志位永远有效
    # .A表示将矩阵转化为列表，nonzero()返回值不为零的元素的下标，[0]表示第一列
    #该行表示读取eCache第一列即是否有效标志位的下标
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:#大于等于2个
        for k in validEcacheList:#在有效标志位中寻找
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):#找到最大步长
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)#随机选择一个j值
        Ej = calcEk(oS, j)#j值对应的误差值Ej
    return j, Ej

'''
功能：更新第k个alpha的误差值至数据结构中
输入：数据集，alpha数
输出：无
'''
def updataEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

'''
功能：完整版Platt SMO内循环，在数据结构中更新alpha数
输入：alpha数，数据集
输出：是否在数据结构中成功更新alpha数，成功返回1，不成功返回0
'''
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)#选择有最大步长的alpha值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        #eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
           #oS.X[j, :] * oS.X[j, :].T
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta >= 0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updataEk(oS, j)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * \
                        (alphaJold - oS.alphas[j])
        updataEk(oS, i)
        #b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
            #oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        #b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             #oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

'''
功能：完整版Platt SMO外循环
输入：数据矩阵dataMatIn，标签向量classLabels，常数C，容错率toler，最大迭代次数maxIter
输出：超平面位移项b，拉格朗日乘子alpha
'''
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)#建立数据结构
    iter = 0#一次迭代完成一次循环过程
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:#判断1
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i: %d, pairs changed %d" % \
                      (iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i: %d, pairs changed %d" % \
                      (iter, i, alphaPairsChanged))
            iter += 1
        # 执行判断1时，如果entireSet = True，表示遍历整个集合，alphaPairsChanged = 0，表示未对任意alpha对进行修改
        if entireSet:
            entireSet = False
        #执行判断1时，第一次迭代遍历整个集合，之后就只遍历非边界值，除非遍历非边界值发现没有任意alpha对进行修改，遍历整个集合
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas

'''
功能：计算超平面法向量
输入：拉格朗日乘子alpha,数据矩阵dataArr，标签向量classLabels
输出：超平面法向量
'''
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

'''
通过核函数将数据转换更高维的空间
Parameters：
    X - 数据矩阵
    A - 单个数据的向量
    kTup - 包含核函数信息的元组
Returns:
    K - 计算的核K
'''
def kernelTrans(X, A, kTup):
    m, n = np.shape(X)
    K = mat(zeros((m,1)))
    if kTup[0] == 'lin': K = X * A.T                       #线性核函数,只进行内积。
    elif kTup[0] == 'rbf':                                 #高斯核函数,根据高斯核函数公式进行计算
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = np.exp(K/(-1*kTup[1]**2))                     #计算高斯核K
    else: raise NameError('核函数无法识别')
    return K