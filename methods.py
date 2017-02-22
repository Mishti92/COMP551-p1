import pandas as pd
import csv 
import numpy as np
import math
import matplotlib
import datetime
import time
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB
threshold = -2


def load_file(prediction_year):
        data = []
        
#         opening File
        with open('Project1_data.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data.append(row)
        return data
    
def split_test_train_data(X,Y,validratio,splitratio):
#     START-- train data --VALIDRATIO-- test data--SPLITRATIO --final test data-- END
    indices = np.random.permutation(X.shape[0])
    split = int(splitratio* X.shape[0])
    valid = int(validratio* X.shape[0])
    trainID, validID, testID = indices[:valid], indices[valid:split], indices[split:]
    X_train, X_valid, X_test = X[trainID], X[validID], X[testID]
    y_train,y_valid,y_test = Y[trainID],Y[validID], Y[testID]
    return X_train, X_valid, X_test,y_train,y_valid, y_test

def value_counts(values):
    uniqDict = {}
    uniqValue = set(list(values))
    for eachEntry in uniqValue:
        uniqDict[eachEntry] = np.sum(values==eachEntry)
    return uniqDict

def featureMeanStd(column):
    sigma = np.std(column)
    mu = np.mean(column)
    return mu, sigma

def pdf_Gaussian_value(x, mu, sigma):
    return (1/(np.sqrt(2*np.pi) * sigma) * np.exp(-((x - mu) ** 2)/(2*sigma ** 2)))

def perf_measure(y_true,y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==1:
            TP += 1
    for i in range(len(y_pred)): 
        if y_pred[i]==1 and y_true[i]==0:
            FP += 1
    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==0:
            TN += 1
    for i in range(len(y_pred)): 
        if y_pred[i]==0 and y_true[i]==1:
            FN += 1
            
    return TP, FP, TN, FN

def accuracy(y_pred,y_true):
#     print 'Accuracy', np.mean(y_pred == y_true)
    
    TP, FP, TN, FN = perf_measure(y_true, y_pred)
    print TP, FP, TN, FN
    accuracy = float(TP+ TN) / (TP + FP + FN + TN)
    precision = float(TP) / (TP+ FP)
    recall = float(TP) / (TP + FN)
    f1= 2*(precision*recall)/(precision+recall)
    
    print 'Accuracy', accuracy
    print 'Precision', precision
    print 'Recall', recall
    print 'F1 Measure', f1
    
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_true,y_pred)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
def process_double_ids(data):
#     for 2016, removing double data for training
        ids = set()
        for id in data:
            years = set()
            for eachEntry in data[id]:
                year = eachEntry[7]
                if year not in years:
                    years.add(year)
                else:
                    ids.add(id)
        for id in ids:
            del data[id]

        return data

def delete_data_above_year(data, year):
# removing data for years above the given year
        for id in data:
            for eachEntry in data[id]:
                entry_year = int(eachEntry[7])
                if entry_year > year:
                    data[id].remove(eachEntry)
        return data
    
def fit(X,y):
    X = np.array(X)
    y = np.array(y)
    Xtrans = X.T
    m, n = X.shape
    count_value = [value_counts(eachItem) for eachItem in Xtrans]
    
#    saving index of different classes in two variables
    positive_class = []
    negative_class = []
    _where_pos = X[y == 1].T
    _where_neg = X[y == 0].T
    
    for index, item in enumerate(count_value):
        # for continuous features
        if len(item) > 2: 
            mu0, sigma0 = featureMeanStd(_where_neg[index])
            mu1, sigma1 = featureMeanStd(_where_pos[index])
            negative_class.append(np.array([mu0, sigma0]))
            positive_class.append(np.array([mu1, sigma1]))
#       for binomial features
        elif len(item) == 2: 
            negative_class.append(value_counts(_where_neg[index])) 
            positive_class.append(value_counts(_where_pos[index]))

#     Total positive and negative         
    total_neg = float(sum(y==0)) 
    total_pos = float(sum(y==1))
    total = total_pos + total_neg
    
#     Calculating Prior Probablities
    prior_prob_pos = total_pos / total
    prior_prob_neg = total_neg / total
    
    return positive_class, negative_class, prior_prob_pos, prior_prob_neg, count_value, total_pos, total_neg

def predict(X_test, positive_class, negative_class, prior_prob_pos, prior_prob_neg,count_value, total_pos, total_neg):
    m, n = X_test.shape
    final_predictions = np.zeros(m)
    X_test = np.array(X_test)
    for i, rows in enumerate(X_test):  # i: sample index
        posProb = np.zeros(n)
        negProb = np.zeros(n)
        
        for j, value in enumerate(rows): # j: feature index
            if type(positive_class[j]) == type(X_test[0]):
                mu0 = negative_class[j][0]
                sigma0 = negative_class[j][1]
                mu1 = positive_class[j][0]
                sigma1 = positive_class[j][1]
                posProb[j] = pdf_Gaussian_value(value, mu1, sigma1)
                negProb[j] = pdf_Gaussian_value(value, mu0, sigma0)
            elif type(positive_class[j] == type(count_value[j])):
#                print negative_class[j]
                n_count = negative_class[j].get(value,0)
                p_count = positive_class[j].get(value,0)
                posProb[j] = (p_count ) / (total_pos * len(positive_class[j]))
                negProb[j] = (n_count ) / (total_neg * len(negative_class[j]))
                
            final_predictions[i] = np.log(prior_prob_pos) + np.sum(np.log(posProb)) - np.log(prior_prob_neg) - np.sum(np.log(negProb))
    
    p = final_predictions
    np.putmask(p, p >= threshold, 1.0)
    np.putmask(p, p < threshold, 0.0)
    return p
        
def process_features(data,prediction_year):        
        featureSet = []
        label = []
        data_by_id = {}

        
        data = data[1:]
        for i in data:
            id = int(i[0])
            if id not in data_by_id:
                data_by_id[id] = []
            data_by_id[id].append(i)

        if prediction_year == 2016:
            data_by_id = process_double_ids(data_by_id) 
        data_by_id = delete_data_above_year(data_by_id, prediction_year)
        
#         print len(data_by_id)
        
        YearBefore = prediction_year - 1
        Year2Before = prediction_year - 2
        Year3Before = prediction_year - 3
        
        for id in data_by_id:
            eachParticipant = data_by_id[id]
            ageSum = timeSum = rankSum = paceSum = 0
            participatedInYearBefore = participatedInYear2Before = participatedInPredictionYear = False
            latestParticipation = totalNumberOfParticipations = 0

            for eachEntry in eachParticipant:
                
                year = int(eachEntry[7])
                if year > prediction_year:
                    continue

                if year == prediction_year:
                    participatedInPredictionYear = True
                    continue
                    
                totalNumberOfParticipations += 1
                ageSum += int(eachEntry[2])
                splitPace = eachEntry[6].split(":")
                paceSum += int(splitPace[0]) * 60 + int(splitPace[1])
                splitTime = eachEntry[5].split(":")
                timeSum += int(splitTime[0]) * 3600 + int(splitTime[1]) * 60 + int(splitTime[2])
                rankSum += int(eachEntry[4])
                participationYear = int(eachEntry[7])
                
                if participationYear > latestParticipation:
                    latestParticipation = participationYear
                    
                if year == YearBefore:
                    participatedInYearBefore = True
                
                if year == Year2Before:
                    participatedInYear2Before = True
                    
                
                
            #Feature Engineering

            if totalNumberOfParticipations >0:
                
                averageTime = float(timeSum) / float(totalNumberOfParticipations)
#                     pace in seconds
                avearagePace = float(paceSum) / float(totalNumberOfParticipations)
                
                averageRank = float(rankSum) / float(totalNumberOfParticipations)
        
                averageAge = float(ageSum) / float(totalNumberOfParticipations)
#                 0 for female and 1 for male
                gender = 0 if (eachParticipant[0][3] == "F") else 1  
#                     time in seconds
                
                participatedInYearBefore = 1 if participatedInYearBefore else 0
                participatedInYear2Before = 1 if participatedInYear2Before else 0
                participatedInPredictionYear = 1 if participatedInPredictionYear else 0
                
                individualFeatureSet = [gender, averageAge, averageTime, averageRank,
                                    float(totalNumberOfParticipations),
                                    participatedInYearBefore
                                    ,participatedInYear2Before
                                   ]
                
#                 Removing values for private runner problem
                if year == 2017:
                    featureSet.append(individualFeatureSet)
                    label.append([id, participatedInPredictionYear])
                else:
                    if id!= 3326:
                        featureSet.append(individualFeatureSet)
                        label.append([id, participatedInPredictionYear])

        return featureSet, label
    
    
def train_on_year(year):
    data = load_file(year)
    featureSet, label = process_features(data,year)
    X = np.array(featureSet)
    y=[]
    for item in label:
        y.append(item[1])
    Y = np.array(y)
    
    trainX, validX, testX, trainY,validY, testY = split_test_train_data(X,Y,0.7,0.9)

    #Validation

    print '****Running Naive Bayes****'
    fit(trainX,trainY)
    pos, neg,  posProb, negProb,count_value, total_pos, total_neg = fit(trainX,trainY)
    yPredict = predict(validX, pos, neg, posProb, negProb,count_value, total_pos, total_neg)
    # print yPredict
    # print min(set(yPredict))
    # print max(set(yPredict))
    # yPredict[41]==1

    accuracy(yPredict,validY)
    
#     Validation

    print 'Validation'
    yPredict = predict(testX, pos, neg, posProb, negProb,count_value, total_pos, total_neg)

    accuracy(yPredict,testY)

    print "Using Gaussian NB SK learn "
    trying_sklearn(trainX,validX,trainY,validY)
    
    return pos, neg, posProb, negProb,count_value, total_pos, total_neg

def predict_for_2017(pos, neg, posProb, negProb,count_value, total_pos, total_neg):
# For 2017
    year = 2017
    data = load_file(year)
    featureSet, label = process_features(data,year)
    
    ids = []
    for item in label:
        ids.append(item[0])


    X_2017 = np.array(featureSet)
    y_naive = predict(X_2017, pos, neg, posProb, negProb,count_value, total_pos, total_neg)
    # len(result)

    predict_file = pd.DataFrame(columns=['PARTICIPANT_ID','Y1_NAIVEBAYES'])


    predict_file.PARTICIPANT_ID= ids
    predict_file.Y1_NAIVEBAYES = y_naive

    print (predict_file['Y1_NAIVEBAYES'].tolist().count(1.0))
    print (predict_file['Y1_NAIVEBAYES'].tolist().count(1.0))*100/len(ids), 'percent'

    predict_file.to_csv('naivebayes_new_2017.csv')
#    print predict_file.head()

def trying_sklearn(trainX,testX,trainY,testY):
    clf = GaussianNB()
    clf.fit(trainX, trainY)
    predictY=clf.predict(testX)
    accuracy(predictY,testY)