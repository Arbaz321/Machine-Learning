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
		# print(self.weights)

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

			#Compute sigmoid
			sigmoid = (float)(1 / (1 + exp(-guess)))

			if sigmoid >= 0.5:
				predictions.append(1)
			else:
				predictions.append(0)
		return predictions

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

def main():
	dataSize = 100
	dataLength = 10
	trainingSize = dataSize
	testSize = dataSize
	
	epochs = 10000
	trials = 5
	average = 0
	
	scaleFeatures = True
	onePerceptron = True

	#initialize one perceptron and keep retraining it
	if onePerceptron:
		p = Perceptron(2)

	for trial in range(1, trials+1):
		#Generate training data for function [y = x]
		features_train = []
		for _ in range(trainingSize):
			x = rand.uniform(-dataLength, dataLength)
			y = rand.uniform(-dataLength, dataLength)
			features_train.append([x, y])
		
		labels_train = []
		for point in features_train:
			if point[1] > point[0]: labels_train.append(1)
			else: labels_train.append(0)

		#feature scaling for training data
		if scaleFeatures:
			features_train = feature_scaling(features_train)
		
		#Generate test data
		features_test = []
		for _ in range(testSize):
			x = rand.uniform(-dataLength, dataLength)
			y = rand.uniform(-dataLength, dataLength)
			features_test.append([x, y])
		
		labels_test = []
		for point in features_test:
			if point[1] > point[0]: labels_test.append(1)
			else: labels_test.append(0)

		#feature scaling for testing data
		if scaleFeatures:
			features_test = feature_scaling(features_test)

		#initialize new perceptron each time
		if not onePerceptron:
			p = Perceptron(len(features_train[0]))

		#fit perceptron
		for _ in range(epochs):
			p.fit(features_train, labels_train)
		
		#predict
		predictions = p.predict(features_test)
		
		#compute accuracy
		score = 0
		for i in range(len(labels_test)):
			if predictions[i] == labels_test[i]:
				score += 1
		average += score / len(labels_test) * 100
		print(trial, ":", score / len(labels_test) * 100)
	print(average / trials)

if __name__ == '__main__':
	main()