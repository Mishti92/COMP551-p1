#packages included
import numpy as np
import matplotlib.pyplot as plt
import math


costArray=np.array([])
iterationCount=0
xLimit=0
yLimit=0

#open input.txt and  read all space separated values
#note:data returned are in string format
def getFileData(fileName):
    data=np.array([])
    fileHandle=open(fileName,'r')
    #print fileHandle
    for line in fileHandle:
        line=line.replace('\r','')
        row=np.array(line.strip().split(' '))
        if data.size==0:
            data=np.append(data,row)
        else:
            data=np.vstack((data,row))
    return data

#split input data and output data
def splitInputOutput(data,sizeInput):
    input=data[:,0:sizeInput]
    #print input
    output=data[:,sizeInput:data.shape[1]]
    #print output
    return (input,output)

#convert string matrices to float
def convertToFloat(X):
    return X.astype(np.float)

def initGraphParameter():
    global interationCount
    global costArray
    global xLimit
    global yLimit
    xLimit=10
    yLimit=0
    interationCount=0
    costArray=np.array([])
    return

#plots graph of cost vs iteration
#expecting a single cost value update
def plotTrainCost(cost):
    global iterationCount
    global costArray
    global xLimit
    global yLimit
    iterationCount+=1
    plt.ion()
    plt.clf()
    plt.grid(True)
    if iterationCount > xLimit:
        xLimit=iterationCount
    if yLimit<cost:
        yLimit=cost

    costArray=np.append(costArray,cost)
    xAxis=np.arange(iterationCount)
    plt.plot(xAxis,costArray)
    plt.axis([0, xLimit, 0, yLimit])
    plt.title('Cost vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.pause(0.01)
    return

#add a row of 1's to the input matrix
#expecting input of the form n*m
def augmentInput(X):
    #n represents number of input vectors
    #m represents number of features
    (n,m)=X.shape
    colOnes=np.ones(n)
    colOnes=np.reshape(colOnes,np.array([n,1]))

    X=np.hstack((X,colOnes))
    #X dimension here would be n*(m+1)
    #print X
    return X

#predict the output for Linear Regression
#expecting coef as (m+1)*1
#expecting input as n*(m+1)
#returning ouput as 1*n
def predictOutputLinear(coef,input,outputFile,write=False):
    output=np.dot(input,coef)
    if write==True:
        np.savetxt(outputFile,output)
    return output

#function to return sigmoid value
def sigmoid(X):
    tmp=np.array([])
    for x in X:
        if tmp.size==0:
            tmp=np.append(tmp,1/(1+math.exp(-x)))
        else:
            tmp=np.hstack((tmp,1/(1+math.exp(-x))))
    return np.reshape(tmp,np.array([X.shape[0],1]))

#def sigmoid(X):
#     return 1/(1+np.exp(-X))


#function to return log value
def log(X):
    tmp=np.array([])
    for x in X:
        if tmp.size==0:
            tmp=np.append(tmp,math.log(x))
        else:
            tmp=np.hstack((tmp,math.log(x)))
    return np.reshape(tmp,np.array([X.shape[0],1]))


#predict the output as per Logistic Regression
#expecting coef as (m+1)*1
#expecting input as n*(m+1)
#returning ouput as 1*n
def predictOutputLogistic(coef,input,outputFile,write=False):
    output=np.dot(input,coef)
    output=sigmoid(output)
    np.putmask(output, output > 0.5, 1.0)
    np.putmask(output, output <= 0.5, 0.0)
    if write == True :
        np.savetxt(outputFile,output)
    return output



#compute MSE cost of given hypothesis over a given set
#expecting coef as (m+1)*1
#expecting costSetIn as n*(m+1)
#expecting costSetOut as n*1
def computeMSECost(costSetIn,costSetOut,coef):
    # n*1
    yPred=np.dot(costSetIn,coef)
    # n*1
    diff=costSetOut-yPred
    cost=np.dot(np.transpose(diff),diff)
    cost=cost/(costSetOut.shape[0]*2)
    return cost

#compute Logistic cost of given hypothesis over a given set
#expecting coef as (m+1)*1
#expecting costSetIn as n*(m+1)
#expecting costSetOut as n*1
def computeCECost(costSetIn,costSetOut,coef):
    # n*1
    yPred=np.dot(costSetIn,coef)
    yPred=sigmoid(yPred)
    # n*1
    p1=np.dot(np.transpose(costSetOut),log(yPred))
    p2=np.dot(np.transpose(1-costSetOut),log(1-yPred))
    cost=p1+p2
    cost=-cost/(costSetOut.shape[0]*2)
    return cost


def computeRegressionAccuracy(costSetIn,costSetOut,coef):
    # n*1
    yPred = np.dot(costSetIn, coef)
    # n*1
    diff = costSetOut - yPred
    #computing the absolute value
    diff=np.absolute(diff)
    avgError=np.average(diff)
    return avgError

def computeClassificationAccuracy(costSetIn,costSetOut,coef):
    # n*1
    yPred = np.dot(costSetIn, coef)
    yPred =sigmoid(yPred)
    np.putmask(yPred, yPred > 0.5, 1.0)
    np.putmask(yPred, yPred <= 0.5, 0.0)
    # n*1
    diff = costSetOut - yPred
    #computing the absolute value
    diff=np.absolute(diff)
    wrongClassifications=np.count_nonzero(diff)
    totalExamples=diff.size
    avgError=float(totalExamples-wrongClassifications)/totalExamples
    tp=0
    tn=0
    fp=0
    fn=0
    for i in range(0,yPred.size):
        if yPred[i]==costSetOut[i] and yPred[i]==1:
            tp+=1
        if yPred[i]==costSetOut[i] and yPred[i]==0:
            tp+=1
        if yPred[i]!=costSetOut[i] and yPred[i]==1:
            fp+=1
        if yPred[i]!=costSetOut[i] and yPred[i] == 0:
            fn += 1
    precision=float(tp)/float(tp+fp)
    recall=float(tp)/float(tp+fn)
    f1=(precision*recall)/(precision+recall)
    print "precision : "+str(precision)
    print "recall : " + str(recall)
    print "f1 : " + str(f1)
    return avgError

#perform linear regression using closed form
#expecting input of the form n*(m+1)
#expecting ouput of the form n*1
def linearRegressionCF(input,output,lambdaValue):
    # n*(m+1)
    X=input
    # (m+1)*n
    Xt=np.transpose(X)
    # (m+1)*(m+1)
    regMat=lambdaValue*np.ones(np.array([Xt.shape[0],Xt.shape[0]]))
    # (m+1)*1
    p2 = np.dot(Xt,output)
    # (m+1)*(m+1)
    p1Internal=np.dot(Xt,X) + (regMat)
    p1=np.linalg.inv(p1Internal)
    # (m+1)*1
    coef=np.dot(p1,p2)
    cost=computeMSECost(input,output,coef)
    acc=computeRegressionAccuracy(input,output,coef)
    return (coef,cost,acc)

#perform linear regression using gradient descent
#expecting input of the form n*(m+1)
#expecting ouput of the form n*1
#threshold defines the improvement below which GD should stop
#all weights are initialiszed with 0
def linearRegressionGD(input,output,threshold,alpha,lambdaValue,graph=False):
    coef=np.zeros(np.array([input.shape[1],1]))
    if graph==True:
        initGraphParameter()
    #just an intialisation value
    costDiff=10
    while costDiff>threshold:
        #print "Running for epoch "+str(epoch)
        #epoch-=1
        # computing simple error for each sample
        # n*1
        err=np.dot(input,coef)-output
        # computing individual updates
        # err_t is 1*n
        # input is n*(m+1)
        # update is 1*(m+1)
        update=np.dot(np.transpose(err),input)
        # transposing update to (m+1)*1
        update=np.transpose(update)
        # performing update due to lambda
        lambdaFact=np.ones(np.array([1,input.shape[1]]))*(1-alpha*lambdaValue/input.shape[0])
        lambdaFact[0,lambdaFact.size-1]=1
        lambdaFact=np.transpose(lambdaFact)
        # performing simultaneous updates of coefficients
        newCoef=np.multiply(lambdaFact,coef)-((alpha/input.shape[0])*update)
        #compute cost and cost difference between iterations
        oldCost=computeMSECost(input,output,coef)
        newCost=computeMSECost(input,output,newCoef)
        costDiff=oldCost-newCost
        #plot the graph
        if graph==True:
            plotTrainCost(newCost)
        #print newCoef
        coef=newCoef

    acc = computeRegressionAccuracy(input, output, coef)
    return (coef,newCost,acc)


#perform logistic regression using gradient descent
#expecting input of the form n*(m+1)
#expecting ouput of the form n*1 and in [0,1]
#threshold defines the improvement below which GD should stop
#all weights are initialiszed with 0
def logisticRegressionGD(input,output,threshold,alpha,lambdaValue,graph=False):
    coef=np.zeros(np.array([input.shape[1],1]))
    if graph==True:
        initGraphParameter()
    #just an intialisation value
    costDiff=10
    coefdiff=1000
    while costDiff>threshold:
        #print "Running for epoch "+str(epoch)
        #epoch-=1
        # computing simple error for each sample
        # n*1
        err=sigmoid(np.dot(input,coef))-output
        # computing individual updates
        # err_t is 1*n
        # input is n*(m+1)
        # update is 1*(m+1)
        update=np.dot(np.transpose(err),input)
        # transposing update to (m+1)*1
        update=np.transpose(update)
        # performing update due to lambda
        lambdaFact=np.ones(np.array([1,input.shape[1]]))*(1-alpha*lambdaValue/input.shape[0])
        lambdaFact[0,lambdaFact.size-1]=1
        lambdaFact=np.transpose(lambdaFact)
        # performing simultaneous updates of coefficients
        newCoef=np.multiply(lambdaFact,coef)-((alpha/input.shape[0])*update)
        #compute cost and cost difference between iterations
        oldCost=computeCECost(input,output,coef)
        newCost=computeCECost(input,output,newCoef)
        costDiff=oldCost-newCost
        coefdiff=np.average(np.absolute(newCoef-coef))
        #plot the graph
        if graph==True:
            plotTrainCost(newCost)
        #print newCoef
        coef=newCoef
    acc = computeClassificationAccuracy(input, output, coef)
    return (coef,newCost,acc)


def runalgo(data,algo,featureCount,threshold,alpha,lambdaValue,graph):

    #reading data from file
    #data=getFileData(inputFile)


    #splitting data into input and output
    (input,output)=splitInputOutput(data,featureCount)
    input=convertToFloat(input)
    output=convertToFloat(output)

    #augmenting the input
    input=augmentInput(input)

    if algo==0:
        #perform linear regression using closed form
        (coef,cost,acc)=linearRegressionCF(input,output,lambdaValue)
        print coef
        print cost
        print acc
        predictOutputLinear(coef, input, "linearResult.txt", True)


    if algo==1:
        #perform linear regression using gradient descent
        (coef,cost,acc)=linearRegressionGD(input,output,threshold,alpha,lambdaValue,graph)
        print coef
        print cost
        print acc
        if graph==True:
            plt.waitforbuttonpress()
        predictOutputLinear(coef, input, "linearResult.txt", True)


    if algo==2:
        (coef,cost,acc)=logisticRegressionGD(input,output,threshold,alpha,lambdaValue,graph)
        print coef
        print cost
        print acc
        if graph==True:
            plt.waitforbuttonpress()
        predictOutputLogistic(coef, input, "logisticResult.txt", True)


    return (coef,cost,acc)

# function to run model on the test data and write results
#return the cost on test data
def testModel(testFile,algo,featureCount,coef):

    # reading data from file
    data = getFileData(testFile)

    # splitting data into input and output
    (input, output) = splitInputOutput(data, featureCount)
    input = convertToFloat(input)
    output = convertToFloat(output)

    # augmenting the input
    input = augmentInput(input)

    if algo==0 or algo==1:
        cost=computeMSECost(input,output,coef)
        acc=computeRegressionAccuracy(input,output,coef)
        predictOutputLinear(coef,input,"linearResult.txt",True)
    elif algo==2:
        cost=computeCECost(input,output,coef)
        acc=computeClassificationAccuracy(input,output,coef)
        predictOutputLogistic(coef,input,"LogisticResult.txt",True)
    return (cost,acc)



#under development . Please ignore this function.
#code to perform cross validation
def crossvalidation(data,folds,algo,featureCount,threshold,alpha,lambdaValue):

    # reading data from file
    errTrain=np.array([])
    errValid=np.array([])
    #data = getFileData(inputFile)
    #randomly shuffle rows
    np.random.shuffle(data)

    #split the data into folds
    dataFolds=np.array_split(data,folds)

    for i in range(folds):

        print "fold:"+str(i)
        #pending
        dataTrain=np.delete(dataFolds,i)
        dataTrain=np.concatenate((dataTrain))

        #dataTrain=np.array([])
        #for j in range(folds):
        #    if j==i:
        #        continue
        #    else:
        #        if dataTrain.size==0:
        #            dataTrain=np.append(dataTrain,dataFolds[j])
        #        else:
        #            dataTrain=np.vstack((dataTrain,dataFolds[j]))

        # pending ends here

        dataValid=dataFolds[i]
        (coef,cost,acc)=runalgo(dataTrain,algo,featureCount,threshold,alpha,lambdaValue,False)
        errTrain=np.append(errTrain,acc)
        #splitting the test set
        (input, output) = splitInputOutput(dataValid, featureCount)
        input = convertToFloat(input)
        output = convertToFloat(output)

        # augmenting the input
        input = augmentInput(input)
        if algo==0 or algo ==1 :
            errValid=np.append(errValid,computeRegressionAccuracy(input,output,coef))
        elif algo==2 :
            errValid = np.append(errValid, computeClassificationAccuracy(input, output, coef))
        #print "avg. training error : " + str(np.average(errTrain))
        #print "avg. validation error : " + str(np.average(errValid))

    print str(lambdaValue)+","+str(np.average(errTrain))+","+str(np.average(errValid))
    #print "avg. training error : "+str(np.average(errTrain))
    #print "avg. validation error : " + str(np.average(errValid))

    return






##########################  MAIN  INTERFACE SINGLE RUNS ###############################
# 0 : linear regression using closed form
# 1 : linear regression using gradient descent
# 2 : logistic regression using gradient descent
#replace training file name with input21.txt
#def runalgo(inputFile,algo,featureCount,threshold,alpha,lambdaValue,graph)

#data=getFileData("App_1_XY_reg_train.txt")
#(coef,cost,acc)=runalgo(data,1,5,5.0,0.01,0.0,False)
#######################################################################################

#used model for prediction

coef=np.array([-0.59584502,-0.45309535,-0.53510437,-1.99446232,1.27347147,-2.07862807])
print testModel("App_1_XY_clas_test.txt",2,5,coef)

coef=np.array([1654.63268758,391.6220372,-92.46059718,18721.9892718,293.0410380,7376.31806956])
print testModel("App_1_XY_reg_test.txt",1,5,coef)
######################################################################################


##########################  MAIN  INTERFACE CROSS VALIDATION #########################
# 0 : linear regression using closed form
# 1 : linear regression using gradient descent
# 2 : logistic regression using gradient descent
#replace training file name with input21.txt
#def crossvalidation(inputFile,folds,algo,featureCount,threshold,alpha,lambdaValue):
#crossvalidation("./Classification/train_2.txt",10,2,5,0.001,4.0,0.0)

#uncomment the following lines(once this)
#print "executing for regression train file 2"
#data=getFileData("App_0_XY_reg_train.txt")
#crossvalidation(data,4,1,4,9.0,0.1,0.0)
#crossvalidation(data,10,1,7,0.1,0.01,0.1)
#crossvalidation(data,10,1,7,0.1,0.01,1.0)
#crossvalidation(data,10,1,7,0.0001,0.01,10.0)
#crossvalidation(data,10,1,7,0.0001,0.01,100.0)

#uncomment the following lines(once this)
#print "executing for classification train file 2"
#data=getFileData("App_1_XY_clas_test.txt")
#print "Lambda,Average Training Error,Average Validation Error"
#crossvalidation(data,3,2,5,0.001,1.0,0.0)
#crossvalidation(data,4,2,5,0.001,4.0,0.1)
#crossvalidation(data,4,2,5,0.001,4.0,1.0)
#crossvalidation(data,4,2,5,0.001,4.0,10.0)
#crossvalidation(data,4,2,5,0.001,4.0,100.0)

######################################################################################




