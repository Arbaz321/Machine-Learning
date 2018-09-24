import matplotlib.pyplot as plt

def graph():
	features_train = load_data('./data/adult_features_train.pickle')
	labels_train = load_data('./data/adult_labels_train.pickle')

	# ages = []
	# for feature in features_train:
	# 	ages.append(feature[0])
	# print(ages, labels_train)

	# # x = range(8)
	# # y = range(8)

	# plt.scatter(ages, labels_train, label='skitscat', color='k', s=25, marker='o')

	# plt.xlabel('age')
	# plt.ylabel('yes or no')
	# plt.title('Test')
	# plt.legend()
	# plt.show()

	num_features = len(features_train[0])
	features_list = []

	for i in range(num_features):
		features_list.append([])

	for entry in features_train:
		for i in range(len(entry)):
			features_list[i].append(entry[i])

	for features in features_list:
		x = features
		y = labels_train
		plt.clf()
		plt.scatter(x, y, label='skitscat', color='k', s=25, marker='o')
		plt.show()
		time.sleep(1)

import pandas as pd
from scipy import stats

def remove_spaces(filename):
	with open(filename, 'r') as file_handler:
		text = file_handler.read().replace(" ", "")
		with open(filename, 'w') as file_handler:
			file_handler.write(text)

def categorize(filename):
	df = pd.read_csv(filename)
	df['weight'] = pd.Categorical(df['weight']).codes
	print(df['weight'])
	with open(filename, 'w') as file_handler:
			df.to_csv(file_handler)

def plot_data(filename):
	with open(filename, 'r') as file_handler:
		df = pd.read_csv(file_handler)
		labels = df['label']
		column_names = df.columns.values
		num_columns = len(column_names)
		for i in range(2, num_columns):
			plt.figure()
			feature = df[column_names[i]]

			slope, intercept, r_value, p_value, std_err = stats.linregress(feature,labels)
			line = slope*feature+intercept

			# plt.scatter(feature, labels, label='skitscat', color='k', s=25, marker='o')
			plt.plot(feature, labels, '-', feature, line)
			plt.xlabel(column_names[i])
			plt.ylabel('label')
		plt.show()
		
import statsmodels.api as sm	
def statsmodel(filename):
	with open(filename, 'r') as file_handler:
		df = pd.read_csv(file_handler)
		X = df[['age', 'weight', 'height']]
		Y = df[['label']]

		X1 = sm.add_constant(X)
		est = sm.OLS(Y, X1).fit()

		est.summary()



def feature_scaling(dataPoints):
	maxFeatures = 0
	minFeatures = 0
	mean = 0
	for point in dataPoints:
		if point > maxFeatures:
			maxFeatures = point
		if point < minFeatures:
			minFeatures = point
		mean += point
	mean /= len(dataPoints)
	for i in range(len(dataPoints)):
		dataPoints[i] = (dataPoints[i] - mean) / (maxFeatures - minFeatures)
	return dataPoints

from sklearn.preprocessing import scale
def scale_features(filename):
	with open(filename, 'r') as file_handler:
		df = pd.read_csv(file_handler)
		ages = list(df['age'])
		print(feature_scaling(ages))

def main():
	file = './test.csv'
	#graph()
	#writeCSV()
	#remove_spaces('./test.csv')
	#categorize('./test.csv')
	#plot_data(file)
	#statsmodel(file)
	scale_features(file)

if __name__ == '__main__':
	main()