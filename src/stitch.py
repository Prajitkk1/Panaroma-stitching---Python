# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:34:10 2020

@author: prajit kk
"""



import numpy as np
import cv2 as cv
import random
import sys
from os import listdir
from scipy.spatial import distance
import os

import warnings
warnings.filterwarnings("ignore")


            #determining matches

def matching_keypoints(kp1, kp2, des1, des2):
    distance_measure = distance.cdist(des1, des2, metric='sqeuclidean')
    threshold  = 10000
    img1 = np.where(distance_measure <= threshold)
    img2 = np.where(distance_measure <= threshold)
    img1 = img1[0]
    img2 = img2[1]
    
    img1_pt = []
    for i in img1:
        img1_pt.append(kp1[i].pt)
        
    img2_pt = []
    for i in img2:
        img2_pt.append(kp2[i].pt)
    img1_pt = np.array(img1_pt)
    img2_pt = np.array(img2_pt)
    return img1_pt,img2_pt


            # determining Homography matrix#
def Homography_matrix(p3, p4):
    p1 = np.array(p3)
    p2 = np.array(p4)
    m1 = []
    p1 = p3
    p2 = p4
    for i in range(len(p1)):
        abcd = [-p1[i][0], -p1[i][1],-1,0,0,0, p1[i][0]*p2[i][0] , p1[i][1]*p2[i][0], p2[i][0] ]
        efgh = [0,0,0,-p1[i][0], -p1[i][1],-1,  p1[i][0] * p2[i][1],  p1[i][1] * p2[i][1], p2[i][1]]
        m1.append(abcd)
        m1.append(efgh)
    m1.append([0,0,0,0,0,0,0,0,1])
    m1 = np.array(m1)
    m2 = np.array([0,0,0,0,0,0,0,0,1])
    try:
        H = np.linalg.solve(m1,m2)
    except:
        return [[1,1,1],[1,1,1],[1,1,1]]
    H = np.reshape(H,(3,3))
    return H
                #ransac algorithm#

def ransac(p3,p4,iterations):
    big_count = 0
    best_H = []
    length = len(p3)

    for i in range(iterations):
        #print(i)
        p1 = []
        p2 = []
        for i in range(4):
            rn = random.randint(0,length-1)
            p1.append(p3[rn])
            p2.append(p4[rn])
        p1 = np.float32(p1)
        p2 = np.float32(p2)
        #H = Homography_matrix(p1,p2)   
        #both function works fine just the below one is faster than my implementation
        H = cv.getPerspectiveTransform(p1,p2)
        rank = np.linalg.matrix_rank(H)
        if(rank<3):
            continue
        point1 = []
        for i in range(length):
            point1.append(np.append(p3[i],1))
        point1 = np.array(point1)
        PH = []
        for i in range(length):
            h_point = np.matmul(H, point1[i])
            PH.append([(h_point[0]/h_point[2]), (h_point[1]/h_point[2])])
        ab_y = p4 - PH
        norm_1 = (np.linalg.norm(ab_y,axis=1)) ** 2
        count = np.where(norm_1 < 0.50)[0]
        inliers = p3[count]
        inlier_count = len(inliers)
        if(inlier_count > big_count):
            big_count = inlier_count
            best_H = H.copy()
    return best_H

            #wrapping the image in both front and reverse way to find the best 
def wrapping(img1,img2,H):
    height1, width1, dims1 = img1.shape
    height2, width2, dims2 = img2.shape
    #print("wrapping started")
    dst1 = cv.warpPerspective(img2,H,(int(width1+width2), int(height1*1.2)))
    dst1[0:img1.shape[0], 0:img1.shape[1]] = img1

    H = np.linalg.inv(H)
    dst2 = cv.warpPerspective(img1,H,(int(width1+width2), int(height1*1.2)))
    dst2[0:img2.shape[0], 0:img2.shape[1]] = img2
    #print("wrapping done")
    return dst1,dst2
    
                    #cropping the black areas#
def cropping(img):
    x = 0
    y = 0
    height, width, dims = img.shape
    for i in range(height):
        for j in range(width):
            if (img[i, j, :] != [0,0,0]).all():
                if j > x:
                   x = j
                if i > y:
                   y = i
    crop_img = img[0:y,0:x, :]
    return crop_img


################################Main ###########################
def main():
    directory = str(sys.argv[1])
    #print("directory",directory)
    onlyfiles = [f for f in listdir(os.path.join(directory)) ]
    
    if "panorama.jpg" in onlyfiles:
        onlyfiles.remove("panorama.jpg")
        #print("removed panroama")
    
    images = []
    for i in onlyfiles:
        images.append(cv.imread(directory+"/"+i))
            #selecting random image at first 
    random_1 = random.randrange(0,len(images))
    img1 = images[random_1]
    first_loop = True
    #images.pop(random_1)
    sift = cv.xfeatures2d.SIFT_create()
    
    while len(images)>0:
        print("image len now :" + str(len(images)))
        high_match_index = 0
        big_match_count = 0
        kps1, desc1 = sift.detectAndCompute(img1,None)
        for i in range(len(images)):
            kps2, desc2 = sift.detectAndCompute(images[i],None)
            mat1,mat2 =  matching_keypoints(kps1,kps2, desc1,desc2)                
    
            if (big_match_count < len(mat1)):
                high_match_index = i
                big_match_count = len(mat1)
            #select the best image with more matches
        if first_loop:
            first_loop = False
            if big_match_count < 5:
                # if matches are less then remove the image
                print("first loop down")
                random_1 = random.randrange(0,len(images))
                img1 = images[random_1]
                continue
        img2 = images[high_match_index]
        try:
            if(np.array(img1)==np.array(img2)).all():
                print("same images")
                images.pop(high_match_index)
                continue
        except:
            pass
        #print("images selected")
        
        height1, width1, dims1 = img1.shape
        height2, width2, dims2 = img2.shape
            
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        mat1,mat2 =  matching_keypoints(kp1,kp2, des1,des2)
        if (len(mat1)< 5):
            print("image has less matches")
            images.pop(high_match_index)
            continue
            
            
        H = ransac(mat2,mat1,1000)
        
        if (H == []):
            #sometimes we get empty matrix, lets delete the image
            del images[high_match_index]
            #print("image no homography")
            continue
        
        dst1,dst2 = wrapping(img1,img2,H)
        
        
        images.pop(high_match_index)
    
                #finding the total zeros in both and selecting the one with less zeros
        unique1, counts1 = np.unique(dst1, return_counts=True)
        unique2, counts2 = np.unique(dst2, return_counts=True)
    
        if(counts1[0]>counts2[0]):
            img1 = dst2
        else:
            img1 = dst1
    
    #plt.imshow(img1)
    
    #print("cropping started")
    final_img = cropping(img1)
    #print("cropping done")
    #plt.imshow(final_img)
    
    cv.imwrite(os.path.join(directory +'/panorama.jpg'),final_img)
    

main()