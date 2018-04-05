from numpy import *

# 创建一个实验样本，以及这些样本的标签;用于训练
def loadDataSet():
    postingList = [ ['my', 'dog', 'has', 'flea', 'problem', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'] ]
    classVec = [0,1,0,1,0,1] # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

# 创建一个包含在所有文档中出现的不重复词的列表，使用set数据集合。
def createVocabList(dataSet):
    vocabSet = set([])  # 定义要返回的向量
    for document in dataSet:    # 遍历文档
        vocabSet = vocabSet | set(document) # 将每个document合并到vocabSet，|用来取两个集合的并集
    return list(vocabSet)   # 把原来的样本全部放入一个集合中，输出列表，就是词汇表

# 输入第一个是词汇表，就是上面那个生成的，第二个是一个文档
'''
词集模型（伯努利NB）
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)    # 初始化一个和词汇表相同大小的向量;
    for word in inputSet:   # 遍历输出集中的单词
        # 如果这个单词在词汇表中，就使返回向量中这个单词在词汇表的位置的那个值为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:   # 否则打印不在列表中
            print('the word: {} is not in my Vocabulary!'.format(word))
    return returnVec     # 返回向量

'''
词袋模型（多项式NB）
'''
def bagOfWords2VecMN(vocalList, inputSet):
    returnVec = [0] * len(vocalList)
    for word in inputSet:
        if word in vocalList:
            returnVec[vocalList.index(word)] += 1
    return returnVec

# 朴素贝叶斯分类器训练函数 (计算条件概率）
def trainNB0(trainMatrix, trainCategory):  # trainMatrix：文档矩阵；trainCategory：每篇文档类别标签所构成的向量
    numTrainDocs = len(trainMatrix) # 获取在测试文档矩阵中有几篇文档
    numWords = len(trainMatrix[0])  # 获取（第一篇）文档的单词长度
    pAbusive = sum(trainCategory) / float(numTrainDocs) # 类别为1的个数除以总篇数，就得到某一类文档在总文档数中所占的比例
                                                        # 计算类别的概率，abusive为1，not abusive为0
    # 初始化求概率的分子变量和分母变量，这里防止有一个p(xn|1)为0，则最后的乘积也为0，所有将分子初始化为1，分母初始化为2。
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 初始化计数器，p1是abusive
    p0Denom = 2.0
    p1Denom = 2.0

    # p0Num = zeros(numWords);p1Num = zeros(numWords)
    # p0Denom = 0.0;p1Denom = 0.0

    for i in range(numTrainDocs):   # 对每一篇训练文档
        if (trainCategory[i] == 1): # 如果这篇文档的类别是1
            p1Num += trainMatrix[i] # 分子就把所有的文档向量按位置累加，trainMatrix[2] = [1,0,1,1,0,0,0];trainMatrix[3] = [1,1,0,0,0,1,1]
            p1Denom += sum(trainMatrix[i]) # 这个是分母，把trainMatrix[2]中的值先加起来为3,再把所有这个类别的向量都这样累加起来，这个是计算单词总数目
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 防止太多的很小的数相乘造成下溢。对乘积取对数.
    # 对每个类别的每个单词的数目除以该类别总数目得条件概率
    p1Vect = log(p1Num / p1Denom)   # change to log()
    p0Vect = log(p0Num / p0Denom)   # change to log()

    # 返回每个类别的条件概率，不是常数，是向量，在向量里面是和词汇表向量长度相同，每个位置代表这个单词在这个类别中的概率
    return p0Vect, p1Vect, pAbusive # 返回两个向量和一个概率

# 朴素贝叶斯分类函数
'''
输入是要分类的向量，使用numpy数组计算两个向量相乘的结果，对应元素相乘，
然后将词汇表中所有值相加，将该值加到类别的对数概率上。
比较分类向量在两个类别中哪个概率大，就属于哪一类.
'''
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

# 测试函数
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    # 把训练文档中所有向量都转换成和词汇表类似的1,0结构
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0v, p1v, pAb = trainNB0(trainMat, listClasses) # 得到了想要的概率

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))  # 把词汇表和测试向量输入
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0v, p1v, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0v, p1v, pAb))

# testingNB()

# 文本解析
# 输入是字符串，输出是单词列表
def textParse(bigString):
    import re   # 导入正则表达式的包
    listOfTokens = re.split('\\W*', bigString)  # 用正则表达式分割字符串
    return [token.lower() for token in listOfTokens if len(token) > 0]  # 用正则表达式分割字符串

# 垃圾邮件测试
def spamTest():
    docList = []    # 定义docList文档列表
    classList = []  # classList类别列表
    fullText = []   # fullText所有文档词汇
    for i in range(1, 26):  # 遍历email/spam和email/ham下的txt文件
        wordList = textParse(open('email/spam/{0}.txt'.format(i)).read())   # 定义并读取垃圾邮件文件的词汇分割列表
        docList.append(wordList)     # 将词汇列表加到文档列表中
        fullText.extend(wordList)    # 将所有词汇列表汇总到fullText中
        classList.append(1)          # 文档类别为1，spam
        wordList = textParse(open('email/ham/{0}.txt'.format(i)).read())    # 读取非垃圾邮件的文档
        docList.append(wordList)    # 添加到文档列表中
        fullText.extend(wordList)   # 添加到所有词汇列表中
        classList.append(0)         # 类别为0，非垃圾邮件
    vocabList = createVocabList(docList)    # 创建词汇列表
    trainingSet = list(range(len(classList)))   # 定义训练集的索引
    testSet = []    # 定义测试集
    for i in range(10): # 随机的选择10个作为测试集
        randIndex = random.randint(0, len(trainingSet)) # 随机整数索引
        testSet.append(trainingSet[randIndex])  # 将随机选择的文档加入到测试集中
        del(trainingSet[randIndex])             # 从训练集中删除随机选择的文档
    trainMat = []   # 定义训练集的矩阵和类别
    trainClasses = []
    for docIndex in trainingSet:    # 遍历训练集，求得先验概率和条件概率
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))   # 将词汇列表变为向量放到trainMat中
        trainClasses.append(classList[docIndex])     # 训练集的类别标签
    p0v, p1v, pSpam = trainNB0(trainMat, trainClasses)  # 计算先验概率，条件概率
    errorCount = 0   # 定义错误计数
    for docIndex in testSet:     # 对测试集进行分类
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])    # 对测试集进行分类
        if classifyNB(wordVector, p0v, p1v, pSpam) != classList[docIndex]:  # 对测试数据进行分类
            errorCount += 1      # 分类不正确，错误计数加1
            if docIndex % 2 == 1:
                print('The error classified email is: spam / {0}.txt'.format((docIndex+1)/2))
            else:
                print('The error classified email is: ham / {0}.txt'.format(docIndex/2))
            print(docList[docIndex])
    print('The error rate is: ', float(errorCount) / len(testSet))  # 输出错误率  

spamTest()






