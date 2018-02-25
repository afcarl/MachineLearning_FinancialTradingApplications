# Mark Trinquero
# MC3 - P1

# Resources Consulted:
# http://quantsoftware.gatech.edu/MC3-Project-1#Hints_.26_resources
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# https://www.youtube.com/watch?v=0fEcg_ZsYNY

import numpy as np
import math


class KNNLearner():

    #Constructor
    def __init__(self, k):
        self.k = k

    #Training Step
    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain, self.Ytrain = Xtrain, Ytrain

    #Query
    def query(self, Xtest, dtype=np.float128):
        Y = np.zeros((Xtest.shape[0],1))

        for i in range(Xtest.shape[0]):
            # Euclidean distance 
            distance =  ((self.Xtrain[:,0] - Xtest[i,0])**2) + ((self.Xtrain[:,1] - Xtest[i,1])**2)
            # Nearest Neighbors
            KNN = [self.Ytrain[nn] for nn in np.argsort(distance)[:self.k]]
            # Take the mean of the closest k points' Y values to make prediction
            Y[i] = np.mean(KNN)
        return Y









































# FOR TESTING
if __name__ == '__main__':

    #inf = open('best4KNN.csv')
    inf = open('Data/ripple.csv')

    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    train_len = int(data.shape[0] * .6)
    train_data, test_data = data[:train_len, :], data[train_len:, :]

    # separate out training and testing data
    Xtrain, Ytrain = train_data[:, :2], train_data[:, 2]
    Xtest, Ytest = test_data[:, :2], test_data[:, 2]

    # create a learner and train it
    learner = KNNLearner(k=99)   # create a KNN learner
    learner.addEvidence(Xtrain, Ytrain) # train it


    # evaluate in sample (Xtrain)
    Y = learner.query(Xtrain) # get the predictions
    predY = [float(item) for item in Y]
    rmse = math.sqrt(((Ytrain - predY) ** 2).sum()/Ytrain.shape[0])

    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytrain)
    print "corr: ", c[0, 1]


    # evaluate out of sample (Xtest)
    Y = learner.query(Xtest) # get the predictions
    predY = [float(item) for item in Y]
    rmse = math.sqrt(((Ytest - predY) ** 2).sum()/Ytest.shape[0])

    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytest)
    print "corr: ", c[0, 1]

