import numpy as np
import matplotlib.pyplot as plt
def loadSimpData():
    data = np.mat([[1.,2.1],
                [2.,1.1],
                [1.3,1.],
                [1.,1.],
                [2.,1.]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return data,classLabels

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curline = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[-1]))
    return  dataMat,labelMat




def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen]<= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return  retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError = D.T*errArr
        #        print('split: dim%d , thresh %.2f, thresh ineqal: %s,the weighted error is %.3f' % (i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] =i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return  bestStump,minError,bestClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)
    m  = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
       # print(aggClassEst)
    return np.sign(aggClassEst)





def adaBoostTrainDS(dataArr,classLabels,numIt= 40):
    weakClassArr= []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
       # print('D:',D.T)
        alpha = float(0.5 * np.log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
      #  print('classEst:',classEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha* classEst
      #  print('aggClassEst:',aggClassEst.T)
        aggError = np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggError.sum()/m
      #  print('total error:',errorRate,'\n')
        if errorRate==0.0:
            break
    return  weakClassArr,aggClassEst

def plotROC(predStrengths,classLabels):
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabels)==1.0)
    yStep =1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax= plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] ==1.0:
            delx = 0
            delY  = yStep
        else:
            delx = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delx],[cur[1],cur[1]-delY],c='b')
        cur = (cur[0]-delx,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for Adaboost  Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print('the Area Under the Curve is : ',ySum*xStep)






if __name__ =='__main__':
    datArr,labelArr = loadDataSet('horseColicTraining2.txt')
    classifierArray,aggClassEst = adaBoostTrainDS(datArr,labelArr,10)
    plotROC(aggClassEst.T,labelArr)

    '''
    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction10 = adaClassify(testArr,classifierArray)
    errArr = np.mat(np.ones((67,1)))
    print(errArr[prediction10!= np.mat(testLabelArr).T].sum())
    '''

'''
    datMat,classLabels = loadSimpData()
    classifierArray = adaBoostTrainDS(datMat, classLabels, 30)
    print(adaClassify([0,0],classifierArray))
'''
'''
    classLabels = np.array(classLabels)
    datMat = np.array(datMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xcord1 = datMat[classLabels==1.0,0]
    ycord1 = datMat[classLabels==1.0,1]
    xcord2 = datMat[classLabels == -1.0,0]
    ycord2 = datMat[classLabels == -1.0,1]
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    ax.legend()
    plt.show()
    D=np.mat(np.ones((5,1))/5)
    bestStump, minError, bestClassEst = buildStump(datMat,classLabels,D)
    print(bestStump,minError,bestClassEst)
'''
