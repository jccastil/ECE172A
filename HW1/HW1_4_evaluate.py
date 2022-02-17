'''
ECE 172A, Homework 1 BONGO
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''

import numpy as np
import matplotlib.pyplot as plt
import random
from skimage import draw

import cv2
import math 

def generate_one_shape(numSides, orientation):
    xCenter = random.randint(200,300)
    yCenter = random.randint(200,300)
    theta = np.linspace(0, 2*np.pi, numSides + 1)
    theta = theta - np.pi/orientation;
    radius = random.randint(40,150)
    x = radius * np.cos(theta) + xCenter;
    y = radius * np.sin(theta) + yCenter;
    binaryImage = poly2mask(x, y, (500, 500))
    return binaryImage

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask

def generator1():
    empty = np.zeros((500,500))
    label = random.randint(0, 1)
    shape = random.randint(3, 4)
    orientation = random.randint(1, 6)
    test_img = generate_one_shape(shape, orientation)
    if label == 0:
        orientation = random.randint(1, 6)
        A = generate_one_shape(shape, orientation)
        B = empty
    else:
        A = empty
        orientation = random.randint(1, 6)
        B = generate_one_shape(shape, orientation)
    return A,B, test_img, label

def generator2():
    label = random.randint(0, 1)
    shape = random.randint(3, 4)
    orientation = random.randint(1, 10)
    test_img = generate_one_shape(shape, orientation)
    orientation = random.randint(1, 10)
    if label == 0:
        A = test_img
        B = generate_one_shape(7-shape, orientation)
    else:
        A = generate_one_shape(7-shape, orientation)
        B = test_img
    return A,B, test_img, label

def generator3():
    label = random.randint(0, 1)
    orientation = random.randint(1, 10)
    test_img = generate_one_shape(3, orientation)
    orientation = random.randint(1, 10)
    if label == 0:
        A = generate_one_shape(3, orientation)
        B = generate_one_shape(4, orientation)
    else:
        A = generate_one_shape(4, orientation)
        B = generate_one_shape(3, orientation)
    return A,B, test_img, label

def generator4():
    label = random.randint(0, 1)
    shape = random.randint(3, 6)
    if label == 0:
        orientation = random.randint(1, 10)
        test_img = generate_one_shape(shape, orientation)
        orientation = random.randint(1, 10)
        A = generate_one_shape(shape, orientation)
        orientation = random.randint(1, 10)
        B = generate_one_shape(9-shape, orientation)
    else:
        orientation = random.randint(1, 10)
        test_img = generate_one_shape(shape, orientation)
        orientation = random.randint(1, 10)
        A = generate_one_shape(9-shape, orientation)
        orientation = random.randint(1, 10)
        B = generate_one_shape(shape, orientation)
    return A,B, test_img, label

def classifier1(A, B, test_img):
    

    return 0

def classifier2(A, B, test_img):
   
    #save the images
    testimg1 = np.uint8(test_img)
    imgA = np.uint8(A)
    imgB = np.uint8(B)

    #save the images as dark
    cv2.imwrite('testimg1.jpg', testimg1)
    cv2.imwrite('imgA.jpg', imgA)
    cv2.imwrite('imgB.jpg', imgB)

    #read the images as grayscale
    testimg1_gs = cv2.imread('testimg1.jpg', cv2.IMREAD_GRAYSCALE)
    imgA_gs = cv2.imread('imgA.jpg', cv2.IMREAD_GRAYSCALE)
    imgB_gs = cv2.imread('imgB.jpg', cv2.IMREAD_GRAYSCALE)

    #turn grayscale images to BW and save them
    cv2.imwrite('testimg1_bw.jpg', testimg1_gs*255)
    cv2.imwrite('imgA_bw.jpg', imgA_gs*255)
    cv2.imwrite('imgB_bw.jpg', imgB_gs*255)


    #binarize the images
    (T,Thresh1) = cv2.threshold(testimg1_gs*255, 128, 255, cv2.THRESH_BINARY)
    (T,Thresh2) = cv2.threshold(imgA_gs*255, 128, 255, cv2.THRESH_BINARY)
    (T,Thresh3) = cv2.threshold(imgB_gs*255, 128, 255, cv2.THRESH_BINARY)

    #test cv matchshapes
    ret_test_A = cv2.matchShapes(Thresh1, Thresh2,1,0.0 )
    print(ret_test_A)
    ret_test_B = cv2.matchShapes(Thresh1, Thresh3,1,0.0 )
    print(ret_test_B)

    if ret_test_A < ret_test_B:
        label == 0
    if ret_test_B < ret_test_A:
        label == 1
    print(label)
    return 0

def classifier3(A, B, test_img):
    """
    YOUR CODE HERE
    """
    return 0

def classifier4(A, B, test_img):
    """
    YOUR CODE HERE
    """
    return 0

for i in range(4):
    correct_count = 0
    for j in range(100):
        if i == 0:
            block0,block1,test_img,label = generator1()
            output = classifier1(block0,block1,test_img)
        elif i == 1:
            block0,block1,test_img,label = generator2()
            output = classifier2(block0,block1,test_img)
        elif i == 2:
            block0,block1,test_img,label = generator3()
            output = classifier3(block0,block1,test_img)
        else: 
            block0,block1,test_img,label = generator4()
            output = classifier4(block0,block1,test_img)
        if output == label:
            correct_count += 1
    accuracy = correct_count / 100
    print('The accuracy of question %d is %f\n'%(i+1,accuracy))