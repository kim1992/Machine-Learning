from numpy import *
import matplotlib.pyplot as plt
import regression as rg

'''
    误差大小评价函数
    Parameters:
        yArr - 真实数据
        yHatArr - 预测数据
    Returns:
        误差大小
'''
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

abX, abY = rg.loadDataSet('abalone.txt')
print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响: ')
yHat01 = rg.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
yHat1 = rg.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
yHat10 = rg.lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
print('k=0.1时,误差大小为:',rssError(abY[0:99], yHat01.T))
print('k=1  时,误差大小为:',rssError(abY[0:99], yHat1.T))
print('k=10 时,误差大小为:',rssError(abY[0:99], yHat10.T))
print('')
print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
yHat01 = rg.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
yHat1 = rg.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
yHat10 = rg.lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
print('k=0.1时,误差大小为:',rssError(abY[100:199], yHat01.T))
print('k=1  时,误差大小为:',rssError(abY[100:199], yHat1.T))
print('k=10 时,误差大小为:',rssError(abY[100:199], yHat10.T))
print('')
print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))

ws = rg.standRegres(abX[0:99], abY[0:99])
yHat = mat(abX[100:199]) * ws
print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))