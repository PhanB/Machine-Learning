import numpy as np
import random
import warnings

class MultilayerPerceptron:
	def __init__(self, input_nodes, hidden_nodes, output_nodes):

		self.num_output_nodes = output_nodes #needed to vectorize classification

		self.learning_rate = 0.05 #learning rate of back propogation

		self.hidden_weights = np.random.rand(hidden_nodes, input_nodes) / 10000 #intialize hidden layer with random weights
		self.output_weights = np.random.rand(output_nodes, hidden_nodes) / 10000 #intialize output layer with random weights


	def sigmoid(self, x):
		return np.float(1.0 / (1.0 + np.exp(-x)))

	def dsigmoid(self, x):
		return np.float(x * (1.0 - x))

	def printWeights(self):
		print(self.hidden_weights)
		print(self.output_weights)

	#convert integer to target vector (i.e. 4 -> [0,0,0,1,0,0,0,0,0])
	def vectorizeTarget(self, classification):
		target = np.zeros(self.num_output_nodes)
		target[classification] = 1
		return target

	def backprop(self, classification, inputs):

		#normalize inputs_list
		inputs = np.array(inputs) / 255.0

		#activation function sigmoid and its derivative (vectorize to be applied to matrices)
		sig = np.vectorize(self.sigmoid, otypes=[np.float])
		dsig = np.vectorize(self.dsigmoid, otypes=[np.float])

		#FEED FORWARD TO GET HIDDEN/NETWORK OUTPUT

		# convert inputs from lists to matrices (dimensions of input_nodes x 1)
		inputs = np.array(inputs, ndmin=2)
		inputs = np.transpose(np.array(inputs, ndmin=2))

		classification_vector = self.vectorizeTarget(classification)
		classification_matrix = np.array(classification_vector, ndmin=2)
		targets = np.transpose(classification_matrix)
		
		#calculate hidden layer
		hidden_inputs = np.dot(self.hidden_weights, inputs)
		hidden_outputs = sig(hidden_inputs)
		
		#calculate network output
		final_inputs = np.dot(self.output_weights, hidden_outputs)
		final_outputs = sig(final_inputs)

		#CALCULATE ERRORS

		#error = y - a = targets - predictions
		output_errors = targets - final_outputs
		
		#hidden error = output_weights * error
		hidden_errors = np.dot(np.transpose(self.output_weights), output_errors) 
		
		#UPDATE WEIGHTS

		#weight update = learning_rate * error * weights
		self.output_weights += self.learning_rate * np.dot((output_errors * dsig(final_outputs)), np.transpose(hidden_outputs))
		
		#weight update = learning_rate * error * weights
		self.hidden_weights += self.learning_rate * np.dot((hidden_errors * dsig(hidden_outputs)), np.transpose(inputs))


	#seperates classification and data (pixel values in this case)
	def parseLine(self, line):
		classification = line[0]
		data = line[1:]
		return classification, data

	def train(self, training_data):

		#run back propogation to adjust weights stoichastically (updating weights for every example)
		for i in range(len(training_data)):
			line = list(training_data.iloc[i])
			classification, data = self.parseLine(line)
			
			self.backprop(classification, data)


	def predict(self, inputs):

		#FEED FORWARD

		sig = np.vectorize(self.sigmoid) #vectorize sigmoid function so it can be applied to all indices in matrix

		#calculate output of hidden layer (sigmoid(W*X), W = weights, X = inputs)
		hidden_inputs = self.hidden_weights.dot(inputs)
		hidden_outputs = sig(hidden_inputs)

		#calculate output of output layer (sigmoid(W*h), W = weights, h = output of hidden layer)
		final_inputs = self.output_weights.dot(hidden_outputs)
		final_outputs = sig(final_inputs)

		#predict one with greatest probability
		return final_outputs.argmax()

		

		

