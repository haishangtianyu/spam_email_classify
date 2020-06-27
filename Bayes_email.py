# -*- coding: utf-8 -*-

import numpy as np
# 正则表达式，处理字符串
import re
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random

# 使用split将字符串拆分为小写字母的单词
def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)  # 用重复任意次的非字符作为切分标志
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 将单词转化为小写

# 载入数据集
def dataload():
    docList = [] # docList：数据集
    classList = [] # classList：标签集
    for i in range(1, 26):# 实际i取1-25的整数值
        wordList = textParse(open('data/spam/%d.txt' % i, 'r').read())  #读取垃圾邮件
        docList.append(wordList)
        classList.append(1)  #1表示垃圾邮件
        wordList = textParse(open('data/ham/%d.txt' % i, 'r').read())  #读取非垃圾邮件
        docList.append(wordList)
        classList.append(0)  #0表示非垃圾邮件
    return docList, classList


# 划分训练集和测试集
def random_split():
    trainingSet = list(range(50)) # 训练集的下标列表
    testSet = []  # 测试集的下标列表
    for i in range(15):  # 7:3随机划分训练集和测试集，随机选取35封邮件作为训练集，15封邮件作为测试集
        randIndex = int(random.uniform(0, len(trainingSet)))# 去掉抽出的测试集下标后从训练集中随机取出一个数字作为测试集的下标
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    return trainingSet, testSet

# 生成词汇表
def createVocabList(docList):# docList：每一封邮件转化成的字符串列表，其中包含的是可能有重复的小写单词
    vocabSet = set([])# vocabSet：不重复的词汇表
    for document in docList:
        vocabSet = vocabSet | set(document)  #集合运算，通过取并集来保证不重复
    return list(vocabSet)

# 将字符串转换为列表，0表示字符在词汇表中不存在，1表示字符在词汇表中存在
def setOfWords2Vec(vocabList, inputSet):# vocabList：词汇表；inputSet：输入的字符串列表
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

#  贝叶斯过程
def trainNB0(trainMat, trainClasses):# trainMat：训练文档矩阵；trainClasses：训练类别标签矩阵
    numTrainDocs = len(trainMat) # 训练邮件的个数
    numWords = len(trainMat[0])
    pAbusive = sum(trainClasses) / float(numTrainDocs)# 因为垃圾邮件标签值为1，非垃圾邮件标签值为0，对标签集数据求和即为垃圾邮件个数，得到垃圾邮件先验概率
    # 单词出现次数初始化为1，避免出现0的情况
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  
    p0Denom = 2.0
    p1Denom = 2.0  # 计算概率时初始化分母为2
    for i in range(numTrainDocs):
        if trainClasses[i] == 1:  
            p1Num += trainMat[i]# 矩阵相加，得到的矩阵表示单词出现情况，p1Num：在每个训练文件中，如果某单词出现过，在对应的地方加一
            p1Denom += sum(trainMat[i])# p1Denom：计算总共出现的不重复的单词数目
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vect = np.log(p1Num / p1Denom)   # 垃圾邮件条件概率
    p0Vect = np.log(p0Num / p0Denom)   # 取对数，便于计算
    return p0Vect, p1Vect, pAbusive

# 比较概率大小进行分类
def classifyNB(vec2Classify, p0Vect, p1Vect, pAbusive):
    p1 = sum(vec2Classify*p1Vect)+np.log(pAbusive)# 计算概率，sum是因为测试文件中单词出现取1，不出现取0，直接乘法，可以免去判断单词是否存在的问题
    p0 = sum(vec2Classify*p0Vect)+np.log(1.0-pAbusive)
    if p1 > p0:
        return 1
    else:
        return 0

#函数功能：绘制混淆矩阵
def cm_plot(confusionMatrix):
    cm = confusionMatrix
    plt.matshow(cm, cmap = plt.cm.Reds)
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('Real label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    plt.show()

#主函数
def main_():
    docList, classList = dataload()
    vocabList = createVocabList(docList)
    trainingSet, testSet = random_split()
    trainMat = []
    trainClasses = [] 
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  # 利用词袋模型得到训练矩阵
        trainClasses.append(classList[docIndex])  # 增加标签
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 进行贝叶斯过程
    Pre_Result = []# 预测标签
    test_true = []# 实际标签
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  # 用词袋模型转化测试集
        Pre_Result.append(classifyNB(np.array(wordVector), p0V, p1V, pSpam))
        test_true.append(classList[docIndex])
    confusionMatrix = confusion_matrix(test_true, Pre_Result) #得到混淆矩阵
    #cm_plot(confusionMatrix)
    accuracy_rate = (confusionMatrix[0][0] + confusionMatrix[1][1]) / len(testSet)
    return accuracy_rate

# 多次训练求正确率平均值
if __name__ == '__main__':
    accuracy_rate = []
    times = int(input("请输入训练次数:\n"))
    for i in range(times):
        accuracy_rate.append(main_())
    accuracy_rate = np.array(accuracy_rate)
    print('垃圾邮件识别准确率: %.2f' %(accuracy_rate.mean()))