import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import pickle

# Clean Data 
def write_csv(filename, content):
	with open(filename, 'w') as file_handler:
		file_handler.write(content)

def remove_spaces(filename, file_handler):
	file = file_handler.read().replace(" ", "")
	write_csv(filename, file)

def add_categorical(df, cols):
	for col in cols:
		df[col] = pd.Categorical(df[col]).codes
	return df

def save_csv(filename, cat_df):
	with open(filename, 'w') as file_handler:
		cat_df.to_csv(file_handler)

def clean_data(filename):
	with open(filename, 'r') as file_handler:
		# Remove spaces
		file_no_spaces = remove_spaces(filename, file_handler)
	with open(filename, 'r') as file_handler:
		df = pd.read_csv(file_handler)
		# Add Categorical Representations for String Data
		category_columns = ['work_class', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'label']
		categorical_df = add_categorical(df, category_columns)
		# Save new CSV
		save_csv(filename, categorical_df)

# Plot Features against Labels
def plot_data(filename):
	with open(filename, 'r') as file_handler:
		df = pd.read_csv(file_handler)
		NUM_POINTS = len(df)
		labels = df['label'][:NUM_POINTS]
		column_names = df.columns.values
		num_columns = len(column_names)
		for i in range(1, num_columns):
			plt.figure()
			feature = df[column_names[i]][:NUM_POINTS]

			slope, intercept, r_value, p_value, std_err = stats.linregress(feature,labels)
			line = slope*feature+intercept

			print(column_names[i], ':', r2_score(labels, line))

			plt.plot(feature, labels, 'None', feature, line)
			plt.xlabel(column_names[i])
			plt.ylabel('label')
		plt.show()
			

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

def main():
	# Clean Data
	CLEAN = False
	if CLEAN:
		files = ['./data/adult_training_data.csv', './data/adult_testing_data.csv']
		for file in files:
			clean_data(file)

	# Plot Data
	PLOT_DATA = True
	if PLOT_DATA:
		file = './data/adult_training_data.csv'
		plot_data(file)

if __name__ == '__main__':
	main()