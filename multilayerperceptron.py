import numpy as np
import math

class MultilayerPerceptron:
	def __init__(self, input_nodes, hidden_nodes, output_nodes):
		self.hidden_weights = np.random.rand(hidden_nodes, input_nodes) #intialize hidden layer with random weights
		self.output_weights = np.random.rand(output_nodes, hidden_nodes) #intialize output layer with random weights

		self.hidden_bias = np.random.rand(hidden_nodes) #intialize hidden bias with random biases 
		self.output_bias = np.random.rand(output_nodes) #intialize output bias with random biases

	def sigmoid(self, x):
		return 1.0 / (1.0 + math.exp(-x))

	def printWeights(self):
		print(self.hidden_weights)
		print(self.output_weights)

	def feedforward(self, inputs):

		sig = np.vectorize(self.sigmoid) #vectorize sigmoid function so it can be applied to all indices in matrix

		#calculate output of hidden layer (sigmoid(WX + B), W = weights, X = inputs, B = hidden bias)
		h = self.hidden_weights.dot(inputs)
		h = np.add(h, self.hidden_bias)
		h = sig(h)

		#calculate output of output layer (sigmoid(Wh + B), W = weights, h = output of hidden layer, B = output bias)
		o = self.output_weights.dot(h)
		o = np.add(o, self.output_bias)
		o = sig(o)

		return o.tolist()