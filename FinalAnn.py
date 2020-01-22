import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') #use 'TkAgg' on local machine & 'Agg' on BlueBear
import matplotlib.pyplot as plt
from sklearn import preprocessing
from random import shuffle
from scipy.stats import wilcoxon
import csv

DataList =[]
AllData = pd.DataFrame() 
featData = pd.DataFrame()
labelData = pd.DataFrame()
originalLabels = pd.DataFrame()
mean_predictions_Q1, my_predictions_Q1 = [], []
mean_predictions_Q2, my_predictions_Q2 = [], []
mean_predictions_Q3, my_predictions_Q3 = [], []
mean_predictions_Q4, my_predictions_Q4 = [], []
mean_predictions_Q5, my_predictions_Q5 = [], []
ftse_pri = np.matrix(pd.read_csv("AllCompanyData/ftse350_ri.csv", skipinitialspace = True, usecols = [1], skiprows = [101], dtype = np.float64).values)

inputs = 13
hidden_neurons = 5
outputs = 1

learning_rate = 0.0000075
epochs = 1000

#initialising the weights and biasses
w1 = tf.Variable(tf.truncated_normal([inputs, hidden_neurons], mean=0.0, stddev=1.0, dtype=tf.float64))
b1 = tf.Variable(tf.constant(0, shape=[hidden_neurons], dtype = tf.float64))
w2 = tf.Variable(tf.truncated_normal([hidden_neurons, outputs], mean=0.0, stddev=1.0, dtype=tf.float64))
b2 = tf.Variable(tf.constant(0, shape=[outputs], dtype = tf.float64))

#TensorFlow placeholder to feed in the data sample by sample
X = tf.placeholder(tf.float64)
Y = tf.placeholder(tf.float64)

def load_data():
	global AllData, DataList, originalLabels
	good_data = pd.read_csv("AllCompanyData/GoodData.csv", skipinitialspace = True, usecols = [1], skiprows = 0, nrows = 70).values
	for file in good_data:
		fileAd = ("AllCompanyData/" + ''.join(file))
		data = pd.read_csv(fileAd, skipinitialspace = True, usecols = [1,2,3,4,5,6,7,8,9,13,14,15,16,17], skiprows = [101], dtype = np.float64).values
		DataList.append(pd.DataFrame(data))
	AllData = pd.concat(DataList)
	AllData.sort_index(inplace = True)
	originalLabels = AllData[10].copy()
	scale_data()
	split_data()

# We scale the data, quarter by quarter, feature by feature in order to avoid data in different quarters affecting each other.
def scale_data(): 
	global AllData
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	for row in xrange(0, 100):
		for col in xrange(0, len(AllData.columns)):
			AllData.loc[row][col] = scaler.fit_transform(AllData.loc[row][col].values.reshape(-1,1))

def split_data():
	global featData, labelData, AllData
	featData = AllData.copy()
	featData.drop(featData.columns[10], axis = 1, inplace = True)
	labelData = AllData[10].copy()
	labelData.values.reshape(-1,1)

def denormalize(normalized, count):
	scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
	scaler.fit(originalLabels.loc[count].values.reshape(-1,1))
	normalized = scaler.inverse_transform(normalized)
	return normalized

def plot_predictions():
	global mean_predictions_Q1, mean_predictions_Q2, mean_predictions_Q3, mean_predictions_Q4, mean_predictions_Q5, my_predictions_Q1, my_predictions_Q2, my_predictions_Q3, my_predictions_Q4, my_predictions_Q5
	to_plot = [mean_predictions_Q1, mean_predictions_Q2, mean_predictions_Q3, mean_predictions_Q4, mean_predictions_Q5, 
	my_predictions_Q1, my_predictions_Q2, my_predictions_Q3, my_predictions_Q4, my_predictions_Q5]
	for i in xrange(0, 5):
		plt.plot(np.matrix(to_plot[i]).reshape(-1,1), label = 'Actual Peformance')
		plt.plot(np.matrix(to_plot[i+5]).reshape(-1,1), label = 'Predicted Performance')
		plt.xlabel('Quarter')
		plt.ylabel('Mean Percentage Change in Return Index')
		plt.legend(loc='best')
		if i == 0:
			plt.title('Best Predicted Performing Portfolio vs. Portfolios Actual Performance')
			plt.savefig('quintile1.png', bbox_inches='tight')
		elif i == 1:
			plt.title('Second Best Predicted Performing Portfolio vs. Portfolios Actual Performance')
			plt.savefig('quintile2.png', bbox_inches='tight')
		elif i == 2:
			plt.title('Third Best Predicted Performing Portfolio vs. Portfolios Actual Performance')
			plt.savefig('quintile3.png', bbox_inches='tight')
		elif i == 3:
			plt.title('Foruth Best Predicted Performing Portfolio vs. Portfolios Actual Performance')
			plt.savefig('quintile4.png', bbox_inches='tight')
		else:
			plt.title('Worst Predicted Performing Portfolio vs. Portfolios Actual Performance')
			plt.savefig('quintile5.png', bbox_inches='tight')
		plt.clf()

def plot_quarters():
	global mean_predictions_Q1, mean_predictions_Q2, mean_predictions_Q3, mean_predictions_Q4, mean_predictions_Q5, ftse_pri
	to_plot = [mean_predictions_Q1, mean_predictions_Q2, mean_predictions_Q3, mean_predictions_Q4, mean_predictions_Q5]
	for i in xrange(0, 5):
		plt.plot(np.matrix(to_plot[i]).reshape(-1,1), label = 'Quintile Actual Peformance')
		plt.plot(ftse_pri, label = 'FTSE 350 Percentage change in RI')
		plt.xlabel('Quarterly Sample')
		plt.ylabel('Mean Percentage Change in Return Index')
		plt.legend(loc='best')
		if i == 0:
			plt.title('Actual Performance Of Top Quintile vs. Market Average Performance')
			plt.savefig('quintile1_actual.png', bbox_inches='tight')
		elif i == 1:
			plt.title('Actual Performance Of Second Best Quintile vs. Market Average Performance')
			plt.savefig('quintile2_actual.png', bbox_inches='tight')
		elif i == 2:
			plt.title('Actual Performance Of Third Best Quintile vs. Market Average Performance')
			plt.savefig('quintile3_actual.png', bbox_inches='tight')
		elif i == 3:
			plt.title('Actual Performance Of Fourth Best Quintile vs. Market Average Performance')
			plt.savefig('quintile4_actual.png', bbox_inches='tight')
		else:
			plt.title('Actual Performance Of Fifth Best Quintile vs. Market Average Performance')
			plt.savefig('quintile5_actual.png', bbox_inches='tight')
		plt.clf()	

def plot_final():
	global mean_predictions_Q1, mean_predictions_Q2, mean_predictions_Q3, mean_predictions_Q4, mean_predictions_Q5, ftse_pri
	plt.plot(np.matrix(mean_predictions_Q1).reshape(-1,1), label = 'Quintile 1 chosen companies')
	plt.plot(np.matrix(mean_predictions_Q2).reshape(-1,1), label = 'Quintile 2 chosen companies')
	plt.plot(np.matrix(mean_predictions_Q3).reshape(-1,1), label = 'Quintile 3 chosen companies')
	plt.plot(np.matrix(mean_predictions_Q4).reshape(-1,1), label = 'Quintile 4 chosen companies')
	plt.plot(np.matrix(mean_predictions_Q5).reshape(-1,1), label = 'Quintile 5 chosen companies')
	plt.plot(ftse_pri, label = 'FTSE 350 Percentage change in RI')
	plt.legend(loc='best')
	plt.title('The FTSE 350 mean percentage change in Return Index versus the chosen companies actual mean performance')
	plt.xlabel('Quarterly Sample')
	plt.ylabel('Percentage Change in Return Index')
	plt.savefig('final.png', bbox_inches='tight')
	plt.clf()

def prediction_handler(myPredictions, actual, count):
	myPredictions = denormalize(myPredictions, count)
	actual = denormalize(np.matrix(actual).reshape(-1,1), count)
	myPredictions = pd.DataFrame(myPredictions)
	actual = pd.DataFrame(actual)
	portfolio_chooser(myPredictions, actual)	

def portfolio_chooser(predictions, actual):
	global mean_predictions_Q1, mean_predictions_Q2, mean_predictions_Q3, mean_predictions_Q4, mean_predictions_Q5, my_predictions_Q1, my_predictions_Q2, my_predictions_Q3, my_predictions_Q4, my_predictions_Q5
	actual_changes, predicted_changes = [], []
	while predictions.empty == False:
		temp = predictions.idxmin()
		actual_changes.append(actual.get_value(temp[0], 0))
		predicted_changes.append(predictions.get_value(temp[0], 0))
		predictions.drop(temp, inplace = True)
	actual_changes = np.matrix(actual_changes).reshape(-1,1)
	mean_predictions_Q5.append((actual_changes[0:14]).mean())
	mean_predictions_Q4.append((actual_changes[14:28]).mean())
	mean_predictions_Q3.append((actual_changes[28:42]).mean())
	mean_predictions_Q2.append((actual_changes[42:56]).mean())
	mean_predictions_Q1.append((actual_changes[-14:]).mean())
	predicted_changes = np.matrix(predicted_changes).reshape(-1,1)
	my_predictions_Q5.append((predicted_changes[0:14]).mean())
	my_predictions_Q4.append((predicted_changes[14:28]).mean())
	my_predictions_Q3.append((predicted_changes[28:42]).mean())
	my_predictions_Q2.append((predicted_changes[42:56]).mean())
	my_predictions_Q1.append((predicted_changes[-14:]).mean())

def wilcoxon_test():
	global mean_predictions_Q1, mean_predictions_Q5
	z_statistic, p_value = wilcoxon(mean_predictions_Q1, mean_predictions_Q5)
	print 'statistic: '
	print z_statistic
	print 'p value:'
	print p_value

def analysis():
	global mean_predictions_Q1, mean_predictions_Q5
	plt.plot(np.matrix(mean_predictions_Q1).reshape(-1,1), label = 'Top Quintile Actual Performance')
	plt.plot(np.matrix(mean_predictions_Q5).reshape(-1,1), label = 'Bottom Quintile Actual Performance')
	plt.legend(loc='best')
	plt.title('The Actual Performances of the Top Quinitle Portfolio vs the Bottom Quintile Portfolio')
	plt.xlabel('Quarterly Sample')
	plt.ylabel('Percentage Change in Return Index')
	plt.savefig('FirstVsFifth.png', bbox_inches='tight')
	plt.clf()
#	differences, absDiff = [], []
#	for (i,j) in (mean_predictions_Q1, mean_predictions_Q5):
#		differences.append((i - j))
#		absDiff.append(np.absolute(i-j))
#	print 'Mean of Differences: '
#	print np.matrix(differences).mean()
#	print 'Median of Differences: '
#	print np.median(differences)
#	print 'Mean of Absolute Differences: '
#	print np.matrix(absDiff).mean()
#	print 'Median of Absolute Differences: '
#	print np.median(absDiff)
#	print 'The Differences: '
#	print differences
#	print 'The Absolute Differences: '
#	print absDiff

def output_csv():
	global mean_predictions_Q1, mean_predictions_Q5
	together = np.column_stack((mean_predictions_Q1, mean_predictions_Q5))
	df1 = pd.DataFrame(together)
	df1.to_csv('ActualPredictions.csv', index = False, header = False)

def create_model(X, w1, w2, b1, b2):
	#the calculations
	w1d = tf.matmul(X, w1)
	h1 = tf.nn.tanh(tf.add(w1d, b1))
	h1w2 = tf.matmul(h1, w2)
	activation = tf.nn.sigmoid(tf.add(h1w2, b2))
	return activation

load_data()


activation = create_model(X, w1, w2, b1, b2)
error = tf.reduce_mean(tf.squared_difference(activation, Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(activation,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.Session() as sess:
	sess.run(init)
	# count corresponds to each quarter
	for count in xrange(0, 99): 
		print "Training round: " + str(count)
		train_ft = np.matrix(featData.loc[count])
		train_labels = np.matrix(labelData.loc[count]).reshape(-1,1)
		pred_ft = featData.loc[count+1]
		pred_labels = labelData.loc[count+1]
		pred_ft = np.matrix(featData.loc[count+1])
		pred_labels = np.matrix(labelData.loc[count+1]).reshape(-1,1)
		for epoch in xrange(epochs):
			for (x,y) in zip(train_ft, train_labels):
				sess.run(optimizer, feed_dict = {X: x, Y: y})		
			if epoch % 2000 == 0:
				print("Training cost = ", sess.run(error, feed_dict={X: featData, Y: labelData}))
		print ("Optimization for round " + str(count) + " finished!")
		print ("Test Data:")
		print ("Training Cost= ", sess.run(error, feed_dict={X: pred_ft, Y: pred_labels}))
		print ("Prediction data:")
		print(accuracy.eval(feed_dict={X: pred_ft, Y: pred_labels}))
		myPredictions = (sess.run(activation, feed_dict={X: pred_ft, Y: pred_labels}))
		prediction_handler(myPredictions, labelData.loc[count+1], count)
	plot_predictions()
	plot_quarters()
	plot_final()
	analysis()
	wilcoxon_test()
	output_csv()

