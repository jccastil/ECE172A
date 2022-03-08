#Test push to github
import numpy as np
import matplotlib as plt
import cv2
import sys

np.set_printoptions(threshold=sys.maxsize)


#Create function computeNormGrayHistogram
def computeNormGrayHistogram():
    #load the image and display it BGR
    img1 = cv2.imread(r"C:\Users\juanc\Desktop\Winter 2022\ECE 172\Github\Homework\ECE172A\HW3\forest.jpg")
    cv2.imshow('test BGR',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Convert image from BGR to RGB
    img1rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    cv2.imshow('test RGB',img1rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Converting the image to gray
    grayimg1 = cv2.cvtColor(img1rgb, cv2.COLOR_RGB2GRAY)
    cv2.imshow('test GRAY',grayimg1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Reshape the image (500x750) into a vector of 375000 elements
    grayimg1 = np.reshape(grayimg1,-1)

    #ThrowawayCode***********************************************
    #Create a 32 length vector to hold counts of intensity values
    #keep in mind initialized to zero may need to be corrected
    # bins = np.zeros(32)
    # for i in range(500):
    #     for j in range(750):
    #         if(0 <= grayimg1[i,j] <= 8):
    #             bins[0] = bins[0] + 1
    # print(bins[0])
    # print(bins)
    #ThrowawayCode***********************************************

    #BIN Method
    #Create 32 bins
    bins32 = np.array(range(0,255,8))
    # print("bins32 is: ",bins32)
    # print("bins32 len is: ", len(bins32))
    #Create an array that shows which bin number the corresponding value in the index of grayimg1 belongs to
    bin_indices = np.digitize(grayimg1, bins32)
    counts = np.bincount(bin_indices)
    #code on next line was necessary in order to see the true counts in each of the bins
    np.set_printoptions(threshold=sys.maxsize)
    print("counts is: ", counts)
    print("counts len: ",len(counts))
    print("sum of counts: ",np.sum(counts))
#call the function
computeNormGrayHistogram()

