import random as rd
import numpy as np
from scipy.spatial import distance
from sklearn.datasets import load_iris

class RandomClassifier():

    def fit(self, data, target):
        self.data = data
        self.target = target
        
    def predict(self, data):
        predictions = []
        for point in data:
            predictions.append(rd.choice(self.target))
        return predictions
        
class NearestNeighboursClassifier():

    def fit(self, data, target):
        self.data = data
        self.target = target
        
    def predict(self, data):
        predictions = []
        for point in data:
            predictions.append(self.closest(point))
        return predictions
        
    def closest(self, point):
        best_index = 0
        best_distance = distance.euclidean(point, self.data[0])
        for i in range(len(self.data)):
            dist = distance.euclidean(point, self.data[i])
            if dist < best_distance:
                best_distance = dist
                best_index = i
        return self.target[best_index]
    
def train_test_split(data,target,test_proportion=0.5):
    training_data, training_target, testing_data, testing_target = [],[],[],[]
    for i in range(len(data)):
        if rd.random()>test_proportion:
            training_data.append(data[i])
            training_target.append(target[i])
        else:
            testing_data.append(data[i])
            testing_target.append(target[i])
    return training_data, testing_data, training_target, testing_target

def accuracy_score(target, predictions):
    return sum(target[i]==predictions[i] for i in range(len(target)))/len(target)
    
iris = load_iris()

train_data,test_data,train_target,test_target = train_test_split(iris.data,iris.target)

randomclassifier = RandomClassifier()

randomclassifier.fit(train_data,train_target)

predictions = myclassifier.predict(test_data)

print(accuracy_score(test_target,predictions))

NNclassifier = NearestNeighboursClassifier()

NNclassifier.fit(train_data,train_target)

predictions = NNclassifier.predict(test_data)

print(accuracy_score(test_target,predictions))