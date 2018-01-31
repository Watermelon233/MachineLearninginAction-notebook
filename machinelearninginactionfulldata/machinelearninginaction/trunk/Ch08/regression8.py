import numpy as np
import matplotlib.pyplot as plt

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


def standRegres(xArr,yArr):
    xMat =np.mat(xArr)
    yMat =np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) ==0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T *yMat )
    return  ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint -xMat[j,:]
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) ==0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T*(weights * yMat))
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()

def ridgeRegres(xMat,yMat,lam =0.2):
    xTx =xMat.T * xMat
    denom = xTx +np.eye(np.shape(xMat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)
    yMat = yMat -yMean
    xMean = np.mean(xMat,0)
    xVar = np.var(xMat,0)
    xMat = (xMat-xMean)/xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:] = ws.T
    return  wMat

def stageWise(xArr,yArr,eps=0.01,numIt = 100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)
    yMat = yMat -yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMean) / xVar
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError =rssE
                    wsMax  = wsTest
        ws =wsMax.copy()
        returnMat[i,:] =ws.T
    return  returnMat



if __name__=='__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    '''
    ws = standRegres(xArr,yArr)
    print(ws)
    xCopy = xArr.copy()
    xCopy.sort()
    yHat = xCopy*ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xMat = np.mat(xArr)[:,1].T
    yMat = np.array(yArr)
    xMat = xMat.tolist()[0]

    ax.scatter(xMat,yMat)
    ax.plot(np.sort(xMat),yHat)
    plt.show()
    '''
    '''
    yHat = lwlrTest(xArr,xArr,yArr,0.02)
    xMat = np.mat(xArr)
    srtInd =xMat[:,1].argsort(0)
    xSort = xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],np.mat(yArr).T.flatten().A[0],s=2,c = 'red')
    plt.show()
    '''
    abX,abY = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(abX,abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()