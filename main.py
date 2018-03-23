import pandas as pd #csv reading
import sys #systems arguments
import os #checking filepaths

from multilayerperceptron import MultilayerPerceptron

NUM_INPUTS = 784 #number of nodes in input layer
NUM_OUTPUTS = 10 #number of nodes in output layer
NUM_HIDDEN = 150 #number of nodes in hidden layer


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
		print("Unable to read: \'" + str(sys.argv[2]) + "\'")
		return 0

	#TRAINING

	#read training data
	training_data = readFile(sys.argv[1])

	#build multilayer perceptron
	neural_net = MultilayerPerceptron(NUM_INPUTS,NUM_HIDDEN,NUM_OUTPUTS)

	#train the neural net
	neural_net.train(training_data)

	#TESTING 

	#read testing data
	testing_data = readFile(sys.argv[2])

	#predict each image in testing data and print accuracy of whole set
	mismatches = 0
	for n in range(len(testing_data)):
		actual = list(testing_data.iloc[n])[0]
		prediction = neural_net.predict(list(testing_data.iloc[n])[1:])

		if actual != prediction:
			mismatches+= 1

	print("Accuracy:", 1.0 - (mismatches / len(testing_data)))


if __name__ == "__main__":
	main()