import random as rand
from math import exp

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
			#Make prediction
			if sigmoid >= 0.5:
				predictions.append(1)
			else:
				predictions.append(0)
		return predictions

	def accuracy(self, predictions, answers):
		score = 0
		for i in range(len(predictions)):
			if predictions[i] == answers[i]:
				score += 1
		score = score / len(answers) * 100
		return score

import pickle
def pickle_data(filename, num_weights):
	p = Perceptron(num_weights)
	with open(filename, 'wb') as file_handler:
		pickle.dump(p, file_handler)

def main():
	PICKLE = True
	if PICKLE:
		filename = './adult_dataset/data/adult_perceptron.pickle'
		pickle_data(filename, 3)

if __name__ == '__main__':
	main()