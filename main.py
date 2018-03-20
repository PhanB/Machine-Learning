import pandas as pd
import sys
import os
import random

from multilayerperceptron import MultilayerPerceptron

NUM_INPUTS = 784 #number of nodes in input layer
NUM_OUTPUTS = 10 #number of nodes in output layer
NUM_HIDDEN = 15 #number of nodes in hidden layer


#returns file contents as DataFrame
def readFile(filename):
	return pd.read_csv(filename, header=None)







def main():

	#CHECK INPUTS
	if len(sys.argv) < 3:
		print("Usage: python3 main.py [training_data.csv] [testing_data.csv]")

	if not os.path.isfile(sys.argv[1]):
		print("Unable to read: \'" + str(sys.argv[1]) + "\'")
		return 0

	if not os.path.isfile(sys.argv[2]):
		print("Unable to read: \'" + str(sys.argv[1]) + "\'")
		return 0

	#TRAINING

	#read training data
	training_data = readFile(sys.argv[1])

	#build multilayer perceptron
	neural_net = MultilayerPerceptron(NUM_INPUTS,NUM_HIDDEN,NUM_OUTPUTS)
	#print(neural_net.feedforward(list(training_data.iloc[0])[1:]))

	#train on data (adjust weights using backpropogation)


	#TESTING

	#read testing data

	#predict each image in testing data and if prediction is wrong increment error count

if __name__ == "__main__":
	main()