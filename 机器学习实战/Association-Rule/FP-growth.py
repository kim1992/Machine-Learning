class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue   # 节点名称
        self.count = numOccur   # 节点出现的次数
        self.nodeLink = None    # 用于链接相似的元素项
        self.parent = parentNode    # 当前节点的父节点
        self.children = {}      # 用于存放节点的子节点

    def inc(self, numOccur):
        self.count += numOccur   # 对count变量增加给定值

    def disp(self, ind = 1):     # 将树以文本形式显示
        print(" " * ind, self.name, " ", self.count)
        for child in self.children.values():
            child.disp(ind + 1)

'''
    FP树构建函数
    # 使用数据集以及最小支持度作为参数来构建FP树。树构建过程会遍历数据集两次。
'''
def createTree(dataSet, minSup = 1):
    headerTable = {}
    # 第一次遍历扫描数据集并统计每个元素项出现的频度。这些信息被保存在头指针中
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 扫描头指针表删除那些出现次数小于minSup的项
    for k in list(headerTable):
        if headerTable[k] < minSup:
            del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if (len(freqItemSet) == 0):  # 如果所有项都不频繁，无需下一步处理
        return None, None

    # 对头指针表扩展以便可以保存计数值及指向每种类型第一个元素项的指针
    for k in headerTable:   #扩展头指针表，第一个元素保存计数
        headerTable[k] = [headerTable[k], None]

    retTree = treeNode('Null Set', 1, None) # 创建只包含空集合的根节点

    # 第二次遍历数据集，创建FP树
    for tranSet, count in dataSet.items():
        localD = {} # 对一个项集tranSet，记录其中每个元素项的全局频率，用于排序
        for item in tranSet:
            if item in freqItemSet: #只对频繁项集进行排序
                localD[item] = headerTable[item][0] # 注意这个[0]，因为之前加过一个数据项
        # 使用排序后的频率项集对树进行填充
        if len(localD) > 0:
            orderdItems = [v[0] for v in sorted(localD.items(),
                            key = lambda t: t[1], reverse = True)]  #排序
            updateTree(orderdItems, retTree, headerTable, count)    # 更新FP树
    return retTree, headerTable

# 为了让FP树生长，需调用updateTree函数
def updateTree(items, inTree, headerTable, count):
    # 该函数首先测试事务中的第一个元素项是否作为子节点存在
    if items[0] in inTree.children: # 存在则计数增加
        inTree.children[items[0]].inc(count)
    else:    # 不存在则将新建该节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)   #创建一个新节点
        if headerTable[items[0]][1] == None:   # 若原来不存在该类别，更新头指针列表
            headerTable[items[0]][1] = inTree.children[items[0]]    # 头指针表更新以指向新的节点
        else:   # 更新头指针表需要调用函数updateHeader
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        # 添加了首节点，递归添加剩下的节点
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

#节点链接指向树中该元素项的每一个实例。
# 从头指针表的 nodeLink 开始,一直沿着nodeLink直到到达链表末尾
def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(simpDat):
    retDict = {}
    for trans in simpDat:
        retDict[frozenset(trans)] = 1
    return retDict

#从叶子节点回溯到根节点
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

'''
创建前缀路径:
    basePet:  输入的频繁项
    treeNode: 当前FP树种对应的第一个节点
    遍历链表直到到达结尾。每遇到一个元素项都会调用ascendTree()来上溯FP树，并收集所有遇到的元素项的名称。
    该列表返回之后添加到条件模式基字典condPats中
'''
def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)    #寻找当前非空节点的前缀
        if (len(prefixPath) > 1):
            condPats[frozenset(prefixPath[1: ])] = treeNode.count  #将条件模式基添加到字典中
        treeNode = treeNode.nodeLink
    return condPats

# 递归查找频繁项集
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    # 1. 初始化一个空列表preFix表示前缀
    # 2. 初始化一个空列表freqItemList接收生成的频繁项集（作为输出）
    bigL = [v[0] for v in sorted(headerTable.items(), key = lambda t: t[1])]   # basePat（按计数值由小到大）
    for basePat in bigL:
        # 记basePat + preFix为当前频繁项集newFreqSet
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet) # 将newFreqSet添加到freqItemList中
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)  # 计算条件FP树（myCondTree、myHead）
        if myHead != None:  # 当条件FP树不为空时，继续下一步；否则退出递归
            # 以myCondTree、myHead为新的输入，以newFreqSet为新的preFix，外加freqItemList，递归这个过程
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)



'''
实例： 从新闻网站点击流中挖掘新闻报道
'''
parsedDat = [line.split() for line in open('kosarak.dat').readlines()]
initSet = createInitSet(parsedDat)
fpTree, headerTab = createTree(initSet, 100000)
myFreqList = []
mineTree(fpTree, headerTab, 100000, set([]), myFreqList)

