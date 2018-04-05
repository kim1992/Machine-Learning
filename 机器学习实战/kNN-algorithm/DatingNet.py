# -*- coding: utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager, FontProperties

'''
将文本记录转换为想要的数据格式
#输入：文件名字符串
#输出：训练样本矩阵和类标签向量
'''

# 读取文本中的数据
def file2matrix(filename):
    fr = open(filename)  # 打开文件
    arrayOfLines = fr.readlines() # 得到一个元素 为文件每一行的列表
    numberOfLines = len(arrayOfLines) # 得到文件的行数
    returnMat = zeros((numberOfLines, 3)) # 创建返回的数据矩阵
    classLabelVector = []    # 创建类标签
    index = 0                # 为了给返回的数据集方便赋值的自增变量

    for line in arrayOfLines:  # 读取文件的每一行并处理
        listFromLine = line.strip().split('\t') # 将每一行去掉换行符并且以\t为分隔符分隔为列表
        returnMat[index,:] = listFromLine[0:3]  # 将列表中的0、1、2号元素赋给returnMat的每一行

        # -1提取列表中的最后一个元素并将其标为int变量 必须明确的通知解释器 否则python语言会将这些元素做字符串处理
        classLabelVector.append(int(listFromLine[-1]))

        index += 1  # 索引加1

    return returnMat, classLabelVector  # 返回数据矩阵和对应的类标签

datingDataMat, datingLabels = file2matrix('/Users/jinjingjie/计算机/机器学习实战/machinelearninginaction/Ch02/datingTestSet2.txt')

'''
使用Matplotlib创建散点图
'''
# 显示中文字体
def getChineseFont():
    return FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

fig = plt.figure() # 创建图形

# 第一个子图 是第2个特征和第3个特征的散点图  但是没有颜色标识
ax = fig.add_subplot(221) # 创建2行2列的第一块的子图
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], s = 10)
plt.xlabel(u"x 玩视频游戏所耗时间百分比",fontproperties=getChineseFont())
plt.ylabel(u'y 每周消耗的冰淇淋公升数',fontproperties=getChineseFont())
plt.title(u'图一（2&&3）',fontproperties=getChineseFont())

# 定义三个类别的空列表
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []

# 第二个子图 是第2个特征和第3个特征的散点图
ax = fig.add_subplot(222)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], s = 10)
# 循环获得每个列表中的值
for i in range(len(datingLabels)):
    if datingLabels[i] == 1:  # 不喜欢
        type1_x.append(datingDataMat[i][1])
        type1_y.append(datingDataMat[i][2])

    if datingLabels[i] == 2:  # 魅力一般
        type2_x.append(datingDataMat[i][1])
        type2_y.append(datingDataMat[i][2])

    if datingLabels[i] == 3:  # 极具魅力
        type3_x.append(datingDataMat[i][1])
        type3_y.append(datingDataMat[i][2])

type1 = ax.scatter(type1_x, type1_y, s=5, c='red')
type2 = ax.scatter(type2_x, type2_y, s=10, c='green')
type3 = ax.scatter(type3_x, type3_y, s=15, c='blue')
ax.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力', ), loc=2)#显示图例  1 右上  2左上 3左下 4 右下 逆时针
plt.xlabel(u"x 玩视频游戏所耗时间百分比",fontproperties=getChineseFont())
plt.ylabel(u'y 每周消耗的冰淇淋公升数',fontproperties=getChineseFont())
plt.title(u'图二（2&&3）',fontproperties=getChineseFont())

#第三个子图 是第1个特征和第2个特征的散点图
ax = fig.add_subplot(2,2,3)#代表创建1行1列从上到下的第三块的子图
#循环获得每个列表中的值
for i in range(len(datingLabels)):
    if datingLabels[i] == 1:  # 不喜欢
        type1_x.append(datingDataMat[i][0])
        type1_y.append(datingDataMat[i][1])

    if datingLabels[i] == 2:  # 魅力一般
        type2_x.append(datingDataMat[i][0])
        type2_y.append(datingDataMat[i][1])

    if datingLabels[i] == 3:  # 极具魅力
        type3_x.append(datingDataMat[i][0])
        type3_y.append(datingDataMat[i][1])

type1 = ax.scatter(type1_x, type1_y, s=5, c='red')
type2 = ax.scatter(type2_x, type2_y, s=10, c='green')
type3 = ax.scatter(type3_x, type3_y, s=15, c='blue')
ax.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2)#显示图例  1 右上  2左上 3左下 4 右下 逆时针
plt.xlabel(u'x  每年获取的飞行常客里程数',fontproperties=getChineseFont())
plt.ylabel(u'y  玩视频游戏所耗时间半分比',fontproperties=getChineseFont())
plt.title(u'图三(1&&2)',fontproperties=getChineseFont())

#第四个子图 是第1个特征和第3个特征的散点图
ax = fig.add_subplot(2,2,4)#代表创建1行1列从上到下的第四块的子图
#循环获得每个列表中的值
for i in range(len(datingLabels)):
    if datingLabels[i] == 1:  # 不喜欢
        type1_x.append(datingDataMat[i][0])
        type1_y.append(datingDataMat[i][2])

    if datingLabels[i] == 2:  # 魅力一般
        type2_x.append(datingDataMat[i][0])
        type2_y.append(datingDataMat[i][2])

    if datingLabels[i] == 3:  # 极具魅力
        type3_x.append(datingDataMat[i][0])
        type3_y.append(datingDataMat[i][2])

type1 = ax.scatter(type1_x, type1_y, s=5, c='red')
type2 = ax.scatter(type2_x, type2_y, s=10, c='green')
type3 = ax.scatter(type3_x, type3_y, s=15, c='blue')
ax.legend((type1, type2, type3), (u'不喜欢', u'魅力一般', u'极具魅力'), loc=2)#显示图例  1 右上  2左上 3左下 4 右下 逆时针
#ax.scatter(datingDataMat[:,0],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.xlabel(u'x  每年获取的飞行常客里程数',fontproperties=getChineseFont())
plt.ylabel(u'y  每周消费的冰淇淋公升数',fontproperties=getChineseFont())
plt.title(u'图四(1&&3)')

plt.show()
