import matplotlib.pyplot as plt
import trees

# 定义决策树决策结果的属性，用字典来定义
# 下面的字典定义也可写作： decisionNode = dict(boxstyle="sawtooth",fc="0.8")
# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
decisionNode = {'boxstyle': 'sawtooth', 'fc': '0.8'}  # 定义decision结点的属性
leafNode = {'boxstyle': 'round4', 'fc': '0.8'}         # 定义leaf结点的属性
arrow_args = {'arrowstyle': '<-'}                  # 定义箭头方向

#声明绘制一个节点的函数
'''
annotate是关于一个数据点的文本
nodeTxt:结点显示的内容
centerPt:文本的中心点 即数据点，箭头所在的点
parentPt:指向文本的点
nodeType：是判断结点还是叶子结点
'''

# 绘制结点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy = parentPt, xycoords = 'axes fraction',
                            xytext = centerPt, textcoords = 'axes fraction',
                            va = 'center', ha = 'center',
                            bbox = nodeType, arrowprops = arrow_args)

# 创建绘图
def createPlot():
    fig = plt.figure(1, facecolor='white') # #新建绘画窗口，背景为白色
    fig.clf()           # 把画布清空
    # createPlot.ax1为全局变量，绘制图像的句柄，
    # subplot为定义了一个绘图，111表示figure中的图有1行1列，即1个，最后的1代表第一个图
    createPlot.ax1 = plt.subplot(111, frameon = False)
    plotNode('a decision node', (0.5, 0.1),(0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# 获得叶结点的数目
def getNumLeafs(myTree):
    numLeafs = 0     # 定义叶子结点数目
    treeKeys = list(myTree.keys())   # 得到树所有键的列表
    rootKey = treeKeys[0]            # 取第一个键
    secondDict = myTree[rootKey]     # 得到第一个键所对应的的值

    for key in secondDict:    # 循环遍历secondDict的键
        if (type(secondDict[key]).__name__ == 'dict'):  # 判断该键对应的值是否是字典类型
            numLeafs += getNumLeafs(secondDict[key])    # 若是则使用递归进行计算
        else:
            numLeafs += 1    # 不是则代表当前就是叶子节点，进行加1即可
    return numLeafs          # 返回叶子结点数目

# 获得叶节点的深度
def getTreeDepth(myTree):
    maxDepth = 0    # 声明最大深度并赋值为0
    treeKeys = list(myTree.keys())
    rootKey = treeKeys[0]
    secondDict = myTree[rootKey]

    for key in secondDict:
        if (type(secondDict[key]).__name__  == 'dict'):
            # 当前树的深度等于1加上secondDict的深度，只有当前点为决策树点深度才会加1
            thisDepth = getTreeDepth(secondDict[key]) + 1
        else:
            # 如果secondDict[key]为叶子结点 则将当前树的深度设为1
            thisDepth = 1
        # 如果当前树的深度比最大数的深度
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth     # 返回树的深度


#plotTree函数

#在父子节点间填充文本
'''
cntrPt：子位置坐标
parentPt：父位置坐标
txtString：文本信息
'''
def plotMixText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]  # 文本填充的x坐标
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]  # 文本填充的y坐标
    createPlot.ax1.text(xMid, yMid, txtString)        # 在（xMid,yMid）位置填充txtString文本

#画树的函数
'''
myTree: 要进行绘制的树
parentPt:父位置坐标
nodeTxt:文本内容

以下均为plotTree函数的成员变量 开始时候从上往下看的 竟然不知道这都是啥，看函数务必要先看入口 然后看调用了什么。

plotTree.xOff    ： 绘制的叶子结点横坐标（变量），只需要负责叶子结点即可。
plotTree.yOff    ： 当前绘制纵坐标（变量），可作为叶子节点和判断节点的纵坐标。
plotTree.totalW：  整棵树的叶子节点数（常量）
plotTree.totalD ： 整棵树的深度（常量）
'''
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)  # 求得myTree的叶子的个数  注意这可不是我们之前所说的那颗最大的树 谁调用它谁是myTree
    depth = getTreeDepth(myTree)    # 求得myTree的深度
    treeKeys = list(myTree.keys())
    rootKey = treeKeys[0]
    # 计算子节点的坐标     其计算有一些需要说明白的，代码后重点讲，大家看完再往后看 记做难点①
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 对判断节点进行的绘制其与其父节点之间的文本信息   此处第一个节点与父节点重合（0.5,1.0）的设置 所以会没有效果 也恰好符合题意
    plotMixText(cntrPt, parentPt, nodeTxt)
    plotNode(rootKey, cntrPt, parentPt, decisionNode)   # 绘制子节点
    secondDict = myTree[rootKey]                        # 得到该节点以下的子树
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD # 深度改变 纵坐标进行减一层

    for key in secondDict.keys():   # 循环遍历各个子树的键值
        if (type(secondDict[key]).__name__ == 'dict'):    # 循环遍历各个子树的键值
            plotTree(secondDict[key], cntrPt, str(key)) # 进行递归绘制
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW # 要绘制下一个叶子节点  横坐标加小矩形的长度（见代码前的上图）
            # 要绘制下一个叶子节点  横坐标加小矩形的长度（见代码前的上图）
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 在父子节点之间填充文本信息
            plotMixText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    # 循环结束  递归回溯（此处大家不明白的 可以再复习一下递归） 因为本层中plotTree.yOff是不变的 所以当递归结束应该进行回溯
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

# 声明绘制图像函数，调用绘制节点函数
def createPlot(inTree):
    fig = plt.figure(1, facecolor= 'white') # 声明绘制图像函数，调用绘制节点函数
    fig.clf()   # 清空绘图区
    axprops = dict(xticks = [], yticks = [])    # 定义横纵坐标轴
    # 创建了属性ax1  functionname.attribute的形式是在定义函数的属性，且该属性必须初始化，否则不能进行其他操作。
    createPlot.ax1 = plt.subplot(111, frameon = False, **axprops)   # 创建1行1列新的绘图区 且图绘于第一个区域 frameon表示不绘制坐标轴矩形 定义坐标轴为二维坐标轴
    plotTree.totalW = float(getNumLeafs(inTree))    # 计算树的叶子数
    plotTree.totalD = float(getTreeDepth(inTree))   # 计算树的深度
    plotTree.xOff = -0.5/plotTree.totalW    # 赋值给绘制叶子节点的变量为-0.5/plotTree.totalW  即为难点②
    plotTree.yOff = 1.0                     # 赋值给绘制节点的初始值为1.0
    plotTree(inTree, (0.5, 1.0), '')        # 调用函数plotTree 且开始父节点的位置为（0.5,1.0） 难点③
    plt.show()   # 画图

'''
难点1：
cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2/plotTree.totalW,plotTree.yOff)的由来
该公式用于计算判断节点的坐标，对于纵坐标，则是不需要计算的，最开始在最上边，按照递归的层次减即可，因此每一层直接使用递归前plotTree.yOff值即可，而相对复杂的是横坐标的值。下面对横坐标的值进行推导。

（1）本颗树的叶子节点所占的总长度：float（numLeafs） *  1.0/plotTree.totalW.
代表叶子节点总个数乘以每一个叶子节点所占的长度。
（2）当前节点位置：float（numLeafs） *  1.0/plotTree.totalW/2
因为当前节点总为递归到本层的根节点，总是在中心的位置，故除以2为其的位置。
（3）当前节点校正过的位置：float（numLeafs） *  1.0/plotTree.totalW/2 +1.0/plotTree.totalW/2
因为plotTree.xOff本来应该在原点，但是为了后期计算方便（进行绘制下一个叶子节点横坐标进行直接加上1.0/plotTree.totalW）赋初值为-0.5/plotTree.totalW，所以加上上一个叶子节点的坐标时需要加上1.0/plotTree.totalW/2 。所以最终结果为plotTree.xOff + (1.0 + float(numLeafs))/2/plotTree.totalW

难点2：
plotTree.xOff = -0.5/plotTree.xOff
因为我们希望叶子节点绘制到小矩形的中间位置较为美观，所以我们复制开始位置时候需要将其值减小矩形长度的一半，之后每次进行计算时只要加上一个小矩形的长度1/plotTree.xOff即可。

难点3：
plotTree(inTree,(0.5,1.0),''）
开始绘制根节点时候我们并没有父节点，所以让这个所谓的父节点和根节点进行重合，这样函数就不会产生多余的节点。
'''

ds,lb = trees.createDataSet()
tree = trees.createTree(ds, lb)
createPlot(tree)

# 使用决策树预测隐形眼镜类型
def plotLensesTree(txtString):
    fr = open(txtString)  #打开文件
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]#读入文件内容并且写入列表中
    lensesLabels = ['age','prescript','astigmatic','tearRate'] #记录标签
    lensesTree = trees.createTree(lenses,lensesLabels)          #创建树
    createPlot(lensesTree)                                      #绘图

plotLensesTree('lenses.txt')