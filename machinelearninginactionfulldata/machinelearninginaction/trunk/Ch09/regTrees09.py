

import numpy as np
def loadDataSet(fileName):
    dataMat  = []
    fr = open(fileName)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltline = list(map(float,curline))
        dataMat.append(fltline)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    try:
        mat0 = dataSet[np.nonzero(dataSet[:,feature]>value)[0]]
        mat1 = dataSet[np.nonzero(dataSet[:,feature]<=value)[0]]
    except:
        print(dataSet[:,feature]<=value)
    return mat0,mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tols = ops[0]
    toln = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:
        return None,leafType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            mat0,mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            if (np.shape(mat0)[0]<toln) or (np.shape(mat1)[0]<toln):
                continue
            newS = errType(mat0)+errType(mat1)
            if newS<bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S-bestS)<tols:
        return None,leafType(dataSet)
    mat0,mat1 =binSplitDataSet(dataSet,bestIndex,bestValue)
    if(np.shape(mat0)[0]<toln) or (np.shape(mat1)[0]<toln):
        return  None,leafType(dataSet)
    return bestIndex,bestValue



def creatTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    retTree={}
    retTree['spInd']=feat
    retTree['spVal']=val
    lSet,rSet = binSplitDataSet(dataSet,feat,val)
    retTree['left'] = creatTree(lSet,leafType,errType,ops)
    retTree['right'] = creatTree(rSet, leafType, errType, ops)
    return  retTree

def isTree(obj):
    return  (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] =getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right']/2.0)

def prune(tree,testData):
    if np.shape(testData)[0]==0:
        return getMean(tree)
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:,-1] - tree['left'],2)) +\
            sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(np.power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else: return tree
    else: return tree


if __name__=='__main__':
    myDat = loadDataSet('ex0.txt')
    myMat = np.mat(myDat)
    print(creatTree(myMat))