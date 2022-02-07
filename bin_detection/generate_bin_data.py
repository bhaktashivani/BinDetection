'''
ECE276A WI21 PR1: Color Classification and Recycling Bin Detection
Author: Shivani Bhakta
Edited On: Feb 6th, 2022
Content: This file contains the code for getting data from using the roipoly
'''


import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('qt5agg')
import numpy as np


def DataCollection():
    folder = 'data/training'

    data_list = []
    for filename in os.listdir(folder):
        print(filename)

        # read the training images
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # display the image and use roipoly for labeling
        fig, ax = plt.subplots()
        ax.imshow(img)
        my_roi = RoiPoly(fig=fig, ax=ax, color='r')

        # get the image mask
        try:
            mask = my_roi.get_mask(img)
        except:
            continue

        # display the labeled region and the image mask
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])

        ax1.imshow(img)
        ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
        ax2.imshow(mask)
        plt.show(block=True)

        # Save the mask into a list and append to the list for later iterations
        # print("mask type: ", type(mask))
        # print("mask shape: ", mask.shape)
        # flatten the whatever ndarray that mask is into 1x(shape of mask)
        # mask = mask.flatten('C')
        # data_list.append(mask)
        # np.save('data/labeled/bluebin/' + str(filename[:-4]), mask)
        np.save('data/labeled/Other/' + str(filename[:-4]), mask)

        # np.save('data/labeled/brown/' + str(filename[:-4]), mask)
        # print("Numpy array for mask image " + str(filename[:-4]) + " saved.")


    # datalist to np array
    #     np.save('data/labeled/binblue', data_list)
    # print("data list length: ", len(data_list))

    # plt.savefig('data/labeled' + filename)



if __name__ == '__main__':
    DataCollection()

