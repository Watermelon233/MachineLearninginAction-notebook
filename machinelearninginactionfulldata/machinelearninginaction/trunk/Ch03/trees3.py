import numpy as np
import operator

def calcshannonEnt(dataset):
    # 香农熵计算
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=1
        else:
            labelCounts[currentLabel]+=1


    shannonEnt =0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob*np.log2(prob)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet,axis,value):
    # 将数据集划分开
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    # 选择最好的特征划分数据
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcshannonEnt(dataSet)
    bestinfoGain = 0.0
    bestfeature=-1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy  += prob * calcshannonEnt(subDataSet)
        infoGain = baseEntropy -newEntropy
        if(infoGain>bestinfoGain):
            # 计算最好的信息增益
            bestinfoGain =infoGain
            bestfeature = i
    return bestfeature

def majorityCnt(classList):
    # 返回最多的类别
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def creatTree(dataSet,labels):
    # 算得决策树
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 类别完全相同时停止划分
        return classList[0]
    if len(dataSet) ==1:  # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    # 得到列表包含的所有属性值
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # Python的列表是引用类型，对引用的修改相当于对原列表修改，因此需要重新复制一份原列表
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # 将标签字符串换为索引
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr  = open(filename)
    return pickle.load(fr)


import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle='sawtooth',fc= '0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')

# 绘制带箭头的注解
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    # annotate(注解文字，xy=箭头起点，xytext=箭头终点，va=垂直居中，ha=水平居中，bbox=文字外框类型，arrowprops=箭头类型属性)
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xytext=centerPt,xycoords='axes fraction',va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.2),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

def getNumLeafs(myTree):
    # 获得树的叶节点
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+= getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return  numLeafs

def getTreeDepth(myTree):
    # 获得树的深度
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1+ getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth>maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    # 快速初始化数据
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def plotMidText(cntrPt,parentPt,txtString):
    # 在父子节点中间填充文本信息
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid,txtString)

def plotTree(myTree,parentPt,nodeTxt):
    # 计算宽和高
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff +(1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeTxt)  # 标记子节点属性值
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff -1.0/plotTree.totalD # 减少y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()



if __name__=='__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = creatTree(lenses,lensesLabels)
    print(lensesTree)
    createPlot(lensesTree)