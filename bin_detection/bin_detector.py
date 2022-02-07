'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage import measure
from skimage.measure import label, regionprops

import matplotlib.pyplot as plt; plt.ion()
import os, cv2
from pathlib import Path
import sys
sys.path.append("../")
sys.path.append("../pixel_classification")
# sys.path.append("../pixel_classification/train_pixel_classifier.py")
from pixel_classification.train_pixel_classifier import train_model,softmax
from pixel_classification.pixel_classifier import PixelClassifier

from numpy import genfromtxt
from skimage.color import label2rgb
import matplotlib.patches as mpatches

class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE

		# convert to HSV color space
		X = img.astype(np.float64)/255
		X = np.reshape(img,(img.shape[0]*img.shape[1],3))
		# print("x shape: ", X.shape)

		y = new_classify(X)
		# print("y shape right after classify: ", y.shape)
		# print(np.unique(y))
		y = y.reshape((img.shape[0],img.shape[1]))
		plt.imshow(y,cmap='gray')
		y_segment = np.zeros((img.shape[0],img.shape[1]))
		#place 1 if the pixel in the original image is blue (1)
		y_segment[np.where(y==0)] = 1
		# mask_img = np.uint8(y_segment)

		# y_segment[np.where(y==3)] = 1
		# y_segment[np.where(y==4)] = 1
		# mask_img = y_segment
		# plt.imshow(y_segment,cmap='gray')
		# print(np.unique(y))
		# plt.imshow(mask_img,cmap='gray')

		# YOUR CODE BEFORE THIS LINE
		################################################################
		return y_segment

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed

			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE

		############################# Citation ##############################
		# This code was used from the example code on the scikit-image.org
		# Link: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py

		label_image = label(img)
		props_eroded = regionprops(label_image)

		boxes = []

		for region in regionprops(label_image):

			if region.area >= 100  :
				# print("region.area: ", region.area)
				minr, minc, maxr, maxc = region.bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
										  fill=False, edgecolor='red', linewidth=2)
				boxes.append([minc,minr,maxc,maxr])

		# YOUR CODE BEFORE THIS LINE
		################################################################

		return boxes




def new_classify(X):

		# X = np.hstack((np.ones((len(X),1)),X))
		X = np.append(X,np.ones([len(X),1]),1) #append column of ones for the bias
		# Using weights from a file so I don't have to train again
		# W = genfromtxt('weights_part2.csv', delimiter=',')
		# print("W print: ", W)


		# W = np.array([
		# 	[ -676.46603756,   676.46603756],
		# 	[-1504.24117728,  1504.24117728],
		# 	[ 1412.57662543, -1412.57662543],
		# 	[ -724.2926877,    724.2926877 ]])

		# # W = np.array([
		# 	[-0.75069237,  2.18681251, -0.39787766, -1.03824248],
		# 	[-1.40009813,  0.21305691, -0.05772499,  1.24476621],
		# 	[ 2.16068364, -2.01396594, -3.0790497,   2.93233201],
		# 	[ 0.39566931,  0.09643401,  1.38059361, -1.87269693]])

		y_predit = np.zeros((X.shape[0],X.shape[1]))
		y_predit = np.matmul(X,W)
		y_predit = softmax(y_predit)
		# print("####################### y_predit shape", y_predit.shape)

		# print("sum = ", np.sum(y_predit,axis=1))


		#convert back the y one-hot encoding to original
		y = np.transpose(np.zeros(X.shape[0]))
		# print("y shape ", y.shape)
		# print(y[:100])
		for i in range(y.shape[0]):
			y[i] = y_predit[i].argmax()
		return y

def get_data():
	'''
	Generates X and y data points from the images and masks labeled

	'''
	folder_train = 'data/training/'
	folder_mask = 'data/labeled/'
	# dirName_mask = ['bluebin/', 'brown/', 'green/', 'notbinblue/']
	dirName_mask = ['bluebin/', 'Other/']

	X1 = None
	X2 = None
	# X3 = None
	# X4 = None

	N = 60
	for i in range(1,N):
		# print("******************************** This is ith iteration: ", i)
		filename1 = str(i).zfill(4)
		filename = str(i).zfill(4) + '.jpg'

		# read image
		img = cv2.imread(os.path.join(folder_train,filename))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# print("img shape: ", img.shape)


		# Getting the pixel points for each color set from all masks

		bluebin_mask_file = Path(folder_mask + dirName_mask[0] + filename1 + ".npy")
		if bluebin_mask_file.is_file():
			# print("bluebin exists")
			bluebin_mask = np.load(bluebin_mask_file, allow_pickle = True)
			if X1 is None:
				X1 = img[bluebin_mask]
			else:
				X1 = np.concatenate((X1,img[bluebin_mask]))


		brown_mask_file = Path(folder_mask + dirName_mask[1] + filename1 + ".npy")
		if brown_mask_file.is_file():
			# print("brown exists")
			brown_mask = np.load(folder_mask + dirName_mask[1] + filename1 + ".npy", allow_pickle = True)
			if X2 is None:
				X2 = img[brown_mask]
			else:
				X2 = np.concatenate((X2,img[brown_mask]))


		# green_mask_file = Path(folder_mask + dirName_mask[2] + filename1 + ".npy")
		# if green_mask_file.is_file():
		# 	# print("green exists")
		# 	green_mask = np.load(folder_mask + dirName_mask[2] + filename1 + ".npy", allow_pickle = True)
		# 	if X3 is None:
		# 		X3 = img[green_mask]
		# 	else:
		# 		X3 = np.concatenate((X3,img[green_mask]))
		#
		#
		# notblue_mask_file = Path(folder_mask + dirName_mask[3] + filename1 + ".npy")
		# if notblue_mask_file.is_file():
		# 	# print("not blue exists")
		# 	notbinblue_mask = np.load(folder_mask + dirName_mask[3] + filename1 + ".npy", allow_pickle = True)
		# 	if X4 is None:
		# 		X4 = img[notbinblue_mask]
		# 	else:
		# 		X4 = np.concatenate((X4,img[notbinblue_mask]))
	# y1, y2, y3, y4 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2), np.full(X3.shape[0],3), np.full(X4.shape[0],4)
	y1, y2 = np.full(X1.shape[0],1), np.full(X2.shape[0], 2)

	# X, y = np.concatenate((X1,X2,X3,X4)), np.concatenate((y1,y2,y3,y4))
	X, y = np.concatenate((X1,X2)), np.concatenate((y1,y2))




	# convert to YUV color space
	# X = cv2.cvtColor(X, cv2.COLOR_RGB2YUV)

	# convert to HSV color space
	# X = cv2.cvtColor(X, cv2.COLOR_RGB2HSV)
	# X = X/255 # X.astype(np.float64)/255
	X = X.astype(np.float64)/255


	# print("X1 shape: ", X1.shape)
	# print("X2 shape: ", X2.shape)
	# print("X3 shape: ", X3.shape)
	# print("X4 shape: ", X4.shape)
	# print("X shape: ", X.shape)
	# print("y shape: ", y.shape)

	return X,y

if __name__ == '__main__':

	# display_mask()

	epochs_num = 150
	alpha = 0.0001
	X,y = get_data()
	X = np.append(X,np.ones([len(X),1]),1) #append column of ones for the bias

	##### One-hot encoding #####
	# Y = np.zeros((y.shape[0],4))
	Y = np.zeros((y.shape[0],y.max()))
	Y[np.arange(y.size),y-1] = 1  # n x K

	# print("X 20 value: ", X[:50])
	# print("Y 20 value: ", Y[:50])
	# print("X -20 value: ", X[5000000:5000050])
	# print("Y -20 value: ", Y[5000000:5000050])

	W = train_model(epochs_num,X,Y,alpha)
	print("W shape: ", W.shape)
	np.savetxt("weights_part2.csv", W, delimiter=",")


	# img = cv2.imread(os.path.join( 'data/validation/','0063.jpg'))
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# my_detector = BinDetector()
	# mask_img = my_detector.segment_image(img)
	# fig, ax = plt.subplots()
	# ax.imshow(mask_img)
	# plt.show(block=True)
