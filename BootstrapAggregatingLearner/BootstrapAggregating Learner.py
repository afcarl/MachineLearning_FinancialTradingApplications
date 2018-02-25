# Mark Trinquero
# MC3-P1 

# Resources Consulted:
# https://www.youtube.com/watch?v=0fEcg_ZsYNY
# https://en.wikipedia.org/wiki/Bootstrap_aggregating
# http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.zeros.html
# https://piazza.com/class/idadrtx18nie1?cid=1055
# https://en.wikipedia.org/wiki/Root_mean_square


import numpy as np
import math
import LinRegLearner as lrl
import KNNLearner as knn 

class BagLearner():

    #Constructor
    def __init__(self, learner, bags, kwargs=None, boost=False):
        self.learner, self.bags, self.boost = learner, bags, boost
        if learner == knn.KNNLearner:
            self.kwargs = kwargs["k"]
        else:
            self.kwargs = kwargs


    #Training Step
    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain, self.Ytrain = Xtrain, Ytrain


    #Query
    def query(self, Xtest):
        # set number of samples per bag equal to the size of the training set
        # https://piazza.com/class/idadrtx18nie1?cid=1055
        baggy_size = self.Xtrain.shape[0]
        Xtrain, Ytrain = np.zeros((baggy_size, 2), dtype=np.float128), np.zeros((baggy_size, ), dtype=np.float128)

        baggy_sack = []
        for i in range(self.bags):
            if self.kwargs:
                learner = self.learner(self.kwargs)
            else:
                learner = self.learner()

            random_indexes = np.random.randint(0, self.Xtrain.shape[0], size=baggy_size)
            index = 0

            for i in random_indexes:
                Xtrain[index] = self.Xtrain[i,:]
                Ytrain[index] = self.Ytrain[i]
                index = index + 1

            learner.addEvidence(Xtrain, Ytrain)
            baggy_sack.append(learner.query(Xtest))
        
        output = sum(baggy_sack)/len(baggy_sack)
        output = [float(i) for i in output]
        return output









































# FOR TESTING
if __name__ == "__main__":
    print 'These are not the drones you are looking for'

    #inf = open('best4KNN.csv')
    inf = open('Data/ripple.csv')

    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    train_len = math.floor(data.shape[0] * .6)
    train_data, test_data  = data[:train_len, :], data[train_len:, :]

    # separate out training and testing data
    Xtrain, Ytrain = train_data[:, :2], train_data[:, 2]
    Xtest, Ytest = test_data[:, :2], test_data[:, 2]

    # create a learner and train it
    learner = BagLearner(learner=knn.KNNLearner, kwargs={'k': 3}, bags=50, boost=False) # create a bag learner
    learner.addEvidence(Xtrain, Ytrain) # train it

    # evaluate in sample (Xtrain)
    predY = learner.query(Xtrain) # get the predictions
    rmse = math.sqrt(((Ytrain - predY) ** 2).sum()/Ytrain.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytrain)
    print "corr: ", c[0, 1]

    # evaluate out of sample (Xtest)
    predY = learner.query(Xtest) # get the predictions
    rmse = math.sqrt(((Ytest - predY) ** 2).sum()/Ytest.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=Ytest)
    print "corr: ", c[0, 1]