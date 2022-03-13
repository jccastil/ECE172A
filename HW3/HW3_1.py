#Test push to github
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

np.set_printoptions(threshold=sys.maxsize)

#read forest.jpg
forestjpg = cv2.imread(r"C:\Users\juanc\Desktop\Winter 2022\ECE 172\Github\Homework\ECE172A\HW3\forest.jpg")
#read template_old.jpg
template_old = cv2.imread(r"C:\Users\juanc\Desktop\Winter 2022\ECE 172\Github\Homework\ECE172A\HW3\template_old.jpeg")


#flip the forest image
forestjpgH = cv2.flip(forestjpg,2)
# cv2.imwrite('forestflipped.jpg',forestjpgH)

#Create function computeNormGrayHistogram
def computeNormGrayHistogram(img):
    #load the image and display it BGR
    # img1 = cv2.imread(r"C:\Users\juanc\Desktop\Winter 2022\ECE 172\Github\Homework\ECE172A\HW3\forest.jpg")
    cv2.imshow('test BGR', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Convert image from BGR to RGB
    bgr2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('test RGB',bgr2rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Converting the image to gray
    imggray = cv2.cvtColor(bgr2rgb, cv2.COLOR_RGB2GRAY)
    cv2.imshow('test GRAY',imggray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #Reshape the image (500x750) into a vector of 375000 elements
    imggray = np.reshape(imggray,-1)

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
    bin_indices = np.digitize(imggray, bins32)
    counts = np.bincount(bin_indices)
    #code on next line was necessary in order to see the true counts in each of the bins
    np.set_printoptions(threshold=sys.maxsize)
    print("counts is: ", counts)
    print("counts len: ",len(counts))
    print("sum of counts: ",np.sum(counts))
    #normalize the bin counts
    counts = counts/(np.sum(counts))
    print("normalized counts: ", counts)
    print("new sum of counts: ", np.sum(counts))
    #plot the histogram from the normalized bins
    figgray = plt.figure()
    # ax = figgray.add_axes([0, 0, 1, 1])
    plt.bar(np.arange(1, 34, 1), counts)
    plt.xlabel("Bin")
    plt.ylabel("Normalized Counts")
    plt.title("Gray Histogram")
    plt.show()

    # #OLD*******************************
    # plt.hist(counts)
    # plt.title('Histogram for forest.jpg (GRAY)')
    # plt.xlabel('Normalized Intensity Values (GRAY)')
    # plt.ylabel('Occurences')
    # # plt.xticks(np.arange(0, .05,.001))
    # plt.show()
    # #may need to fix plot so that x axis is bin number
    # #in histogram you are currently displaying "2 times normalized bin count + 4 times normalized bin count etc = 1
    # # OLD*******************************
def computeNormRGBHistogram(img):
    # load the image and display it BGR
    cv2.imshow('test BGR', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #obtain R,G,B channels| Note: will be leaving image in BGR (B==0, G==1, R==2)
    redchannel = img[:,:,2]
    greenchannel = img[:, :, 1]
    bluechannel = img[:, :, 0]

    #test save the images
    #cv2.imwrite(r"C:\Users\juanc\Desktop\Winter 2022\ECE 172\Github\Homework\ECE172A\HW3\frstred.jpg",redchannel)

    #This is strictly for visualizing each color channel******************************
    #make B,G == 0 to visualize red image
    redimg = np.zeros(img.shape)
    redimg[:,:,2] = redchannel
    #cv2.imwrite(r"C:\Users\juanc\Desktop\Winter 2022\ECE 172\Github\Homework\ECE172A\HW3\frstredvis.jpg",redimg)
    # make B,R == 0 to visualize green image
    greenimg = np.zeros(img.shape)
    greenimg[:, :, 1] = greenchannel
    #cv2.imwrite(r"C:\Users\juanc\Desktop\Winter 2022\ECE 172\Github\Homework\ECE172A\HW3\frstgreenvis.jpg", greenimg)
    # make G,R == 0 to visualize blue image
    blueimg = np.zeros(img.shape)
    blueimg[:, :, 0] = bluechannel
    #cv2.imwrite(r"C:\Users\juanc\Desktop\Winter 2022\ECE 172\Github\Homework\ECE172A\HW3\frstbluevis.jpg", blueimg)
    #This is strictly for visualizing each color channel******************************

    #bins process RED
    redchannel = np.reshape(redchannel,-1)
    # redchannel = redchannel*2
    redbins = np.array(range(0,255,8))
    redbin_ind = np.digitize(redchannel, redbins)
    redcounts = np.bincount(redbin_ind)
    # code on next line was necessary in order to see the true counts in each of the bins
    np.set_printoptions(threshold=sys.maxsize)
    print("redcounts is: ", redcounts)
    print("redcounts len: ", len(redcounts))
    print("sum of redcounts: ", np.sum(redcounts))

    #bins process GREEN
    greenchannel = np.reshape(greenchannel, -1)
    greenbins = np.array(range(0, 255, 8))
    greenbin_ind = np.digitize(greenchannel, greenbins)
    greencounts = np.bincount(greenbin_ind)
    # code on next line was necessary in order to see the true counts in each of the bins
    np.set_printoptions(threshold=sys.maxsize)
    print("greencounts is: ", greencounts)
    print("greencounts len: ", len(greencounts))
    print("sum of greencounts: ", np.sum(greencounts))

    #bins process BLUE
    bluechannel = np.reshape(bluechannel, -1)
    bluebins = np.array(range(0, 255, 8))
    bluebin_ind = np.digitize(bluechannel, bluebins)
    bluecounts = np.bincount(bluebin_ind)
    # code on next line was necessary in order to see the true counts in each of the bins
    np.set_printoptions(threshold=sys.maxsize)
    print("bluecounts is: ", bluecounts)
    print("bluecounts len: ", len(bluecounts))
    print("sum of bluecounts: ", np.sum(bluecounts))

    #normalize EACH histogram
    redcounts = redcounts/(np.sum(redcounts))
    greencounts = greencounts/(np.sum(greencounts))
    bluecounts = bluecounts/(np.sum(bluecounts))

    #create color coded concatenated histogram in RGB order (96 elements) (will be 99 until resolved)
    #print the size of each count
    # print("redcountsize: ", len(redcounts))
    # print("greencountsize: ", len(greencounts))
    # print("bluecountsize: ", len(bluecounts))
    #each verified to be 33, first element is always 0 (might remove later)

    #concatenate the counts
    rgbcount = np.concatenate([redcounts,greencounts,bluecounts])
    #verified that it is 99 elements. Must resolve to be 96.

    #plot the histogram of RGB channels (using plt.bar)
    # plot the histogram from the normalized bins
    figgray = plt.figure()
    # ax = figgray.add_axes([0, 0, 1, 1])
    plt.bar(np.arange(1, 100, 1), rgbcount)
    plt.xlabel("Bin")
    plt.ylabel("Normalized Counts")
    plt.title("RGB Histogram")
    plt.show()


#call the functions
#computeNormGrayHistogram(forestjpg)

#computeNormRGBHistogram(forestjpg)
