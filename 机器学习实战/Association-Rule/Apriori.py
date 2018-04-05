def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    # C1是空列表，用来存储所有不重复的项值。如果某个物品项没有在C1中出现，则将其添加到C1中
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not[item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))
    # frozenset是指"冰冻"的集合（不可改变）。
    # 因为之后要将这些集合作为字典的key使用，frozenset可以实现， 而set不可以。

def scanD(D, Ck, minSupport):   # 该函数用于从 C1 生成 L1
    ssCnt = {}
    for tid in D:   #遍历数据集
        for can in Ck:  #遍历候选项
            if can.issubset(tid):    #判断候选项中是否含数据集的一部分（子集）
                if not can in ssCnt:
                    ssCnt[can] = 1  #不含设为1
                else:
                    ssCnt[can] += 1 #有则计数加1
    numItems = float(len(D))    #数据集大小
    retList = []        #L1初始化
    supportData = {}    #记录候选项中各个数据的支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems #计算支持度
        if support >= minSupport:
            retList.append(key)         #满足条件加入L1中
        supportData[key] = support
    return retList, supportData

'''
函数说明： 创建候选项集Ck
参数：     Lk：频繁项集列表
          k：项集元素的个数
'''
def aprioriGen(Lk, k):
    retList = []    # 创建一个空列表
    lenLk = len(Lk) # 计算Lk中的元素
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:    # 当前k-2个项相同时，将两个集合合并
                retList.append(Lk[i] | Lk[j])
    return retList

'''
函数说明：生成候选项集的列表
参数：   dataSet，数据集
        minSupport，支持度
'''
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]    # 将L1放入L列表中
    k = 2
    while (len(L[k - 2]) > 0):  # while循环将L2, L3, L4, ... 放入L列表中，直到下一个大的项集为空
        Ck = aprioriGen(L[k - 2], k)     # 调用aprioriGen()创建候选项集Ck
        Lk, supK = scanD(D, Ck, minSupport) # 扫描数据集，从Ck得到Lk
        supportData.update(supK)    # update()方法将一个字典的内容添加到另一个字典中
        L.append(Lk)
        k += 1
    return L, supportData

'''
关联规则生成函数，此函数调用其他两个函数rulesFromConseq、calcConf
    L: 频繁项集列表
    supportData: 包含那些频繁项集支持数据的字典
    minConf: 最小可信度阈值，默认是0.7
    函数最后要生成一个包含可信度的规则列表，后面可以基于可信度对它们进行排序
    这些规则存放在bigRuleList中。
'''
def generateRules(L, supportData, minConf = 0.7):
    bigRuleList = []    #存储所有的关联规则
    # 注意,i从1开始，表示只取项数大于等于2的项集
    # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]    #该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1): #如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:    # 如果项集中只有两个元素，那么需要使用calcConf()来计算可信度值
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf = 0.7):
    prunedH = []     #返回一个满足最小可信度要求的规则列表
    for conseq in H:    # 遍历H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, "-->", conseq, "conf: ", conf)   #如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
            brl.append((freqSet - conseq, conseq, conf))    #添加到规则里，brl 是前面通过检查的
            prunedH.append(conseq)  # prunedH 保存规则列表的右部
    return prunedH


'''
用于生成候选规则集合，从最初的项集中生成更多的关联规则
        freqSet: 频繁项集
        H: 可以出现在规则右部的元素列表
'''
def rulesFromConseq(freqSet, H, supportData, brl, minConf = 0.7):
    m = len(H[0])   # H 中频繁项集大小m
    # 查看该频繁项集是否大到可以移除大小为m的子集
    if (len(freqSet) > (m + 1)):    #频繁项集元素数目大于单个集合的元素数
        Hmp1 = aprioriGen(H, m + 1) #存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)   #计算可信度
        if (len(Hmp1) > 1): #满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            # 如果满足最小可信度的候选关联规则数目大于1，那么递归
            # 将项数+1，继续进行过滤,直到候选关联规则数目小于等于1或者freqSet数目<=m+1，
            # 例如{1,2,3}不能以{1,2,3}为后件
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

ds = loadDataSet()
l,sd = apriori(ds)
# rules = generateRules(l, sd, 0.75)

def loadMushroom():
    fileName = 'mushroom.dat'
    fr = open(fileName)
    dataMat = []
    for line in fr.readlines():
        curLine = line.split()
        dataMat.append(curLine)
    return dataMat

dm = loadMushroom()
ll, sdd = apriori(dm, 0.3)

# 在结果中搜索包含有毒特征值2的频繁项集

k = 3       # 最关联特征（2个）
for item in ll[k]:
    if item.intersection('2'):
        print(item)



