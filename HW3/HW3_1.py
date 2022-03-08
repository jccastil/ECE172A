#Test push to github
import numpy as np
import matplotlib as plt
import cv2


#Create function computeNormGrayHistogram
def computeNormGrayHistogram():
    #load the image and display it
    img1 = cv2.imread(r"C:\Users\juanc\Desktop\Winter 2022\ECE 172\Github\Homework\ECE172A\HW3\forest.jpg")
    cv2.imshow('test', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#call the function
computeNormGrayHistogram()
