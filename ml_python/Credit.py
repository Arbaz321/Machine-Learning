def getDataFromCSV(size):
	#import data from csv file
	import pandas as pd
	df = pd.read_csv('credit.csv')
	#df['(Name of Column)'] returns list of column values associated with name of col
	#df.loc[(Number of row)] returns list of row values 

	#save half of rows as training features, (labels = last column)
	num_items = df['ID']
	dataSize = size

	features_train = []
	labels_train = []
	trainingSize = dataSize
	for i in range(1, trainingSize):
		item = df.loc[i]
		features = []
		for j in range(1, len(item)-1):
			features.append(item[j])

		labels_train.append(item[len(item)-1])		
		features_train.append(features)

	#save second half as testing features, (labels = last column)
	features_test = []
	labels_test = []
	testingStart = trainingSize
	testingSize = testingStart + trainingSize

	for i in range(testingStart, testingSize):
		item = df.loc[i]
		features = []
		for j in range(1, len(item)-1):
			features.append(item[j])

		labels_test.append(item[len(item)-1])		
		features_test.append(features)
	return features_train, labels_train, features_test, labels_test

def getDataFromPickle():
	import pickle
	with open('pickles/credit_features_train.pickle', 'rb') as fp:
		features_train = pickle.load(fp)
	with open('pickles/credit_labels_train.pickle', 'rb') as fp:
		labels_train = pickle.load(fp)
	with open('pickles/credit_features_test.pickle', 'rb') as fp:
		features_test = pickle.load(fp)
	with open('pickles/credit_labels_test.pickle', 'rb') as fp:
		labels_test = pickle.load(fp)
	return features_train, labels_train, features_test, labels_test

def createPickle(features_train, labels_train, features_test, labels_test):
	import pickle
	with open('pickles/credit_features_train.pickle', 'wb') as fp:
		pickle.dump(features_train, fp)
	with open('pickles/credit_labels_train.pickle', 'wb') as fp:
		pickle.dump(labels_train, fp)
	with open('pickles/credit_features_test.pickle', 'wb') as fp:
		pickle.dump(features_test, fp)
	with open('pickles/credit_labels_test.pickle', 'wb') as fp:
		pickle.dump(labels_test, fp)	

def feature_scaling(dataPoints):
	maxFeatures = []
	minFeatures = []
	mean = []
	for _ in range(len(dataPoints[0])):
		maxFeatures.append(0)
		minFeatures.append(0)
		mean.append(0)
	for point in dataPoints:
		for i in range(len(point)):
			if point[i] > maxFeatures[i]:
				maxFeatures[i] = point[i]
			if point[i] < minFeatures[i]:
				minFeatures[i] = point[i]
			mean[i] += point[i]
	for i in range(len(mean)):
		mean[i] /= len(dataPoints)
	for i in range(len(dataPoints)):
		for j in range(len(mean)):
			dataPoints[i][j] = (dataPoints[i][j] - mean[j]) / (maxFeatures[j] - minFeatures[j])
	return dataPoints

import random as rand
from math import exp
from math import log 

class Perceptron():
	weights = []
	def __init__(self, numWeights):
		self.learningRate = 0.001
		self.weights = []
		for i in range(numWeights):
			self.weights.append(rand.uniform(-1, 1))

	def fit(self, data_features, data_labels):
		newWeights = []
		#Loop through all weights
		for i in range(len(self.weights)):
			sumError = 0
			#Loop through all data points
			for index in range(len(data_features)):
				#Compute guess
				guess = 0
				for j in range(len(self.weights)):
					guess += (self.weights[j] * data_features[index][j])
				#Compute sigmoid
				guess = round(guess, 15)
				# print(guess)
				sigmoid = (float)(1 / (1 + exp(-guess)))
				#Compute error
				error = (sigmoid - data_labels[index]) * data_features[index][i]
				sumError += error
			#Compute new weights
			delta = sumError * self.learningRate / len(data_features)
			newWeight = self.weights[i] - delta
			newWeights.append(newWeight)
		#Simulateously update weights
		self.weights = newWeights
	
	def predict(self, data_features):
		predictions = []
		for point in data_features:
			#Compute guess
			guess = 0
			for i in range(len(self.weights)):
				guess += (self.weights[i] * point[i])
			guess = round(guess, 15)
			# print(guess)	
			#Compute sigmoid
			sigmoid = (float)(1 / (1 + exp(-guess)))

			if sigmoid >= 0.5:
				predictions.append(1)
			else:
				predictions.append(0)
		return predictions

from sklearn.neighbors import KNeighborsClassifier
def main():
	dataSize = 100
	epochs = 100
	
	scaleFeatures = True
	extract_from_CSV = False

	#Getting data
	if extract_from_CSV:
		features_train, labels_train, features_test, labels_test = getDataFromCSV(dataSize)
	else:
		features_train, labels_train, features_test, labels_test = getDataFromPickle()

	#featureScaling
	if scaleFeatures:
		features_train = feature_scaling(features_train)
		features_test = feature_scaling(features_test)
	
	#initialize perceptron
	num_weights = len(features_train[0])
	p = Perceptron(num_weights)
	# p = KNeighborsClassifier()

	#fit perceptron
	for _ in range(epochs):
		p.fit(features_train, labels_train)
	
	# #KNeightborsClassifier Test
	# p = KNeighborsClassifier()
	# p.fit(features_train, labels_train)

	#predict
	predictions = p.predict(features_test)
	# print(predictions)

	#compute accuracy
	score = 0
	for i in range(len(labels_test)):
		if predictions[i] == labels_test[i]:
			score += 1
	print(score / len(labels_test) * 100)

	if extract_from_CSV:
		createPickle(features_train, labels_train, features_test, labels_test)
		
	
if __name__ == '__main__':
	main()