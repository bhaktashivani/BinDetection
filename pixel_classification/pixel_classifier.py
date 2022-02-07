'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''



from __future__ import division
import sys
sys.path.append("../")
sys.path.append("../pixel_classification")
import numpy as np
from numpy import genfromtxt
# from train_pixel_classifier import *
# from generate_rgb_data import read_pixels
from pixel_classification.generate_rgb_data import read_pixels


class PixelClassifier():
	def __init__(self):
		'''
			Initilize your classifier with any parameters and attributes you need
		'''

		self.W = np.array([
					[ 5.11203148, -2.55654972, -2.55548176],
					[-2.46611678,  4.93538393, -2.46926715] ,
					[-2.41749716, -2.47756157,  4.89505874],
					[-0.08038083,  0.04575378,  0.03462705]])

	
	def classify(self,X):
		'''
			Classify a set of pixels into red, green, or blue

			Inputs:
			  X: n x 3 matrix of RGB values
			Outputs:
			  y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		X = np.append(X,np.ones([len(X),1]),1)
			# Using weights from a file so I don't have to train again
		# W = genfromtxt('weights.csv', delimiter=',')
		# print("shape of W: ", W.shape)
		# print("Type of W: ", type(W))
		# print("W: ", W)
		W = self.W

		# y_predit = np.zeros((X.shape[0],3))
		y_predit = np.matmul(X,W)
		y_predit = self.softmax(y_predit)

		#convert back the y one-hot encoding to original
		y = np.transpose(np.zeros(X.shape[0]))
		for i in range(y.shape[0]):
			y[i] = y_predit[i].argmax() + 1

		# YOUR CODE BEFORE THIS LINE
		################################################################
		return y

	def softmax(self,z):
		'''
		Softmax function with input numpy array z - 3694 x 3
		'''

		# using the s(z) = s(z-c1), where c ∈ R is any constant,
		# e.g., C = max(i) * z(i) is useful for numerical conditioning
		c = z.max(axis=1).reshape((-1,1))  #- 3694 x 1
		# print("zshape: ", z.shape)
		# print("c shape: ", c.shape)
		# e = np.ones(z.shape, dtype=int)

		# num = np.exp(z - e*np.transpose(c))
		num = np.exp(z - c) # shape of  (3694, 3)
		denom = np.sum(num, axis = 1, keepdims= True)
		s = num/denom
		return s

if __name__ == '__main__':
	folder = 'data/validation/blue'
	X = read_pixels(folder)
	myPixelClassifier = PixelClassifier()
	myPixelClassifier.classify(X)