'''
	Dataset: https://archive.ics.uci.edu/ml/datasets/Adult
	April Chen: https://github.com/aprilypchen/depy2016/blob/master/DePy_Talk.ipynb
'''
import random as rand
from math import exp
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron

class Perceptron():
	weights = []
	def __init__(self, numWeights):
		self.learningRate = 0.0001
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

def extract_data(filename, columns):
	with open(filename, 'r') as file_handler:
		df = pd.read_csv(filename)
		num_entries = len(df)
		features = []
		labels = list(df['label'])
		for i in range(num_entries):
			entry = []
			for col in columns:
				entry.append(df[col][i])
			features.append(entry)
		return features, labels

def scale_data(data, num_data):
	return scale(data[:num_data])

def main():
	# FEATURES: ['age', 'education_num', 'marital_status', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week'] 
	columns = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
	features_train, labels_train = extract_data('./data/adult_training_data.csv', columns)
	features_test, labels_test = extract_data('./data/adult_testing_data.csv', columns)

	num_data = 100
	# Not scaled
	features_train, labels_train, features_test, labels_test = features_train[:num_data], labels_train[:num_data], features_test[:num_data], labels_test[:num_data]

	# Scaled
	scaled_features_train = scale_data(features_train, num_data)
	scaled_features_test = scale_data(features_test, num_data)

	OTHER_CLASSIFIER = False
	num_weights = len(features_train[0])
	epochs = 10000
	trials = 1
	NEW_PERCEPTRON = True

	average = 0
	avg_trials = 0

	start_time = time.time()
	print('Start:', start_time)
	
	if not OTHER_CLASSIFIER:
		clf = Perceptron(num_weights)
		for trial in range(trials):
			if NEW_PERCEPTRON:
				clf = Perceptron(num_weights)
			for epoch in range(epochs): 
				clf.fit(scaled_features_train, labels_train)
			predictions = clf.predict(scaled_features_test)
			accuracy = clf.accuracy(predictions, labels_test)	
			print(trial + 1, ':', accuracy)
			avg_trials += accuracy
		print(avg_trials // trials)
	else:
		#clf = DecisionTreeClassifier()
		#clf = RandomForestClassifier()
		#clf = LinearRegression(normalize=True)
		#clf = GradientBoostingClassifier()
		clf = Perceptron()
		clf.fit(scaled_features_train, labels_train)
		predictions = clf.predict(scaled_features_test)
		print(accuracy_score(labels_test, predictions)*100)
		# print(clf.score(labels_test, predictions))
			
	end_time = time.time()
	print('End:', end_time)
	print('Time Total:', (end_time - start_time)*1000, 'milliseconds')

if __name__ == '__main__':
	main()