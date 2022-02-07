'''
ECE 276A  WI22 PR1: Color Classification and Recycling Bin Detection
Author: Shivani A Bhakta
Created on: Jan 30, 2022
Content: This file contains the code for the training model for the color classifer.
'''

from __future__ import division
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../pixel_classification")
# from generate_rgb_data import read_pixels
from pixel_classification.generate_rgb_data import read_pixels

def loadData():
    '''
    This function loads the training Data in numpy arrays
    # this code is copy form the starter code in another file in this folder
    output: X - datapoints and y - labels in nparray
    '''

    folder = 'data/training'
    X1 = read_pixels(folder+'/red', verbose = True)
    X2 = read_pixels(folder+'/green')
    X3 = read_pixels(folder+'/blue')
    y1, y2, y3 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3)
    X, y = np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3))
    #X size: (3694, 3)
    #Y size: (3694,)

    return X,y

def softmax(z):
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

def getGradient(W,X,Y):
    '''
    Computes gradient (1 Epoch)
    inputs:
            W: Weights with shape  (4, 3)
            X: Data points matrix (ndarray) with shape  (3694, 4)
            Y: Labels after one-hot encoding with shape
    '''
    # print("---------------- inside getGradient ------------")
    z = np.matmul(X,W)  # n x K
    y_pred = softmax(z)
    # print("my loss: ", -1*np.sum(Y*np.log(y_pred)))
    temp = Y - y_pred
    result = np.matmul(np.transpose(X), temp)

    return result


def train_model(epochs, X, Y, alpha):
    # Logistic Regression Training
    W = np.zeros((X.shape[1], Y.shape[1])) # (K,d) = (3,3) # initalize weights to Zeros
    # print("Size of W: ", W.shape)
    for epoch in range(epochs):
        print("************ This is Epoch: ",  epoch+1,  " ************")
        # MLE Step
        W = W + alpha*getGradient(W,X,Y)

    return W


def train( ):
    # n: # of data points
    # K: # of classes
    # Parameters
    epochs = 50
    alpha = 0.001 #learning rate

    X,y = loadData()
    X = np.append(X,np.ones([len(X),1]),1) #append a column of ones for bias

    ######### One-hot encoding ########
    # y ∈ {1,2,3} --> Y ∈ {(1,0,0), (0,1,0), (0,0,1)}
    Y = np.zeros((y.shape[0],3))
    Y[np.arange(y.size),y-1] = 1 # n x K
    ##################################

    # Logistic Regression Training
    W = train_model(epochs, X, Y, alpha)

    # # Logistic Regression Training
    # W = np.zeros((X.shape[1], Y.shape[1])) # (K,d) = (3,3) # initalize weights to Zeros
    # # print("Size of W: ", W.shape)
    # for epoch in range(epochs):
    #     print("************ This is Epoch: ",  epoch+1,  " ************")
    #     # MLE Step
    #     W = W + alpha*getGradient(W,X,Y)


    # print("Size of W: ", W.shape)
    return W


if __name__ == '__main__':

    #save the weights for later use
    #parameters

    W = train()
    # save weights in a CSV file
    np.savetxt("weights.csv", W, delimiter = ",") # https://stackoverflow.com/questions/3345336/save-results-to-csv-file-with-python