# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:14:33 2020

@author: OS
"""

#from __future__ import print_function

import numpy as np
import random
from functools import reduce
import matplotlib.pyplot as plt
import scipy.optimize as optim

def creat_weight(n):
    random.seed(1213)
    nums = [random.uniform(0,1) for x in range(0,n)]
    summary = reduce(lambda x,y: x+y, nums)
    norm = [x/summary for x in nums]
    weight_set = np.array(norm)
    return weight_set

#Lp distance between two points
def get_distance(a, b, p):
    distance = np.linalg.norm(a-b, ord=p)
    return distance

#objective function
def weighed_distance(theta, x_y, w):
    #for i in range(len(x_y)):
        #dist = np.sum(w[i] * np.linalg.norm(x_y[i]-theta[i], ord=k))
    #x_y = np.column_stack((x,y))
    num = x_y.shape[0]
    dist = 0.0
    for i in range(num):
        distmat = np.linalg.norm(x_y[i]-theta, ord=2)
        distmat = w[i]*distmat
        dist = dist + distmat
    return dist

#两个数组元素相乘求和
def sumAndMul2List(list1, list2):

    result = np.sum([a*b for a,b in zip(list1,list2)])

    return result

#Optimizing the center points with BFGS
def kmeans(samples, k, cutoff):
    #samples_xy = samples[:,0:2].tolist()
    #clusters = random.sample(samples_xy, k)
    clusters = np.array([[0,0],[5,5]])
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        lists_w = [[] for _ in range(k)]
        for sample in samples:
            smallest_distance = get_distance(sample[0:2], clusters[0],2)
            cluster_index = 0
            
            for i in range(k-1):
                distance = get_distance(sample[0:2], clusters[i+1],2)
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1
                    
            lists_xy[cluster_index].append(sample[0:2])
            lists_w[cluster_index].append(sample[2])
            
        biggest_shift = 0.0
        for j in range(k):
            #print(len(lists_xy[j]))
            lists_xy[j] = np.array(lists_xy[j])
            lists_w[j] = np.array(lists_w[j])
            j_num = np.sum(lists_w[j])
            result_x = sumAndMul2List(lists_w[j], lists_xy[j][:,0])
            clusters_x = result_x/j_num
            result_y = sumAndMul2List(lists_w[j], lists_xy[j][:,1])
            clusters_y = result_y/j_num

            spare_clusters = np.append(spare_clusters, [[clusters_x, clusters_y]], axis=0)
            shift = get_distance(clusters[j], [clusters_x, clusters_y],2)
            biggest_shift = max(biggest_shift, shift)
        #print(biggest_shift)
        if biggest_shift < cutoff:
            #print("第{}次迭代后，聚类稳定。".format(n_loop))
            break
        else:
            clusters = spare_clusters
            #print(clusters)
            n_loop += 1
            #print(lists_w)
    return clusters, lists_xy, lists_w

#Sorting to determine the weighted median
def w_median(points, w_value):
    Z = zip(points, w_value)
    Z = sorted(Z)
    points_sorted, w_value_sorted = zip(*Z)
    points_sorted = np.array(points_sorted)
    w_value_sorted = np.array(w_value_sorted)
    return points_sorted, w_value_sorted

#Sum of the previous subarrays
def part_sum(len_k, w_sorted):
    p_sum = 0
    for i in range(len_k):
        p_sum += w_sorted[i]
        if p_sum > 0.50000000000000000 or p_sum == 0.50000000000000000:
            return i
        else:
            continue
     
#Optimizing the center points with Manhattan
def manhattan(samples, k, cutoff):
    #samples_xy = samples[:,0:2].tolist()
    #clusters = random.sample(samples_xy, k)
    clusters = np.array([[0,0],[5,5]])
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        lists_w = [[] for _ in range(k)]
        for sample in samples:
            smallest_distance = get_distance(sample[0:2], clusters[0], 1)
            cluster_index = 0
            
            for i in range(k-1):
                distance = get_distance(sample[0:2], clusters[i+1], 1)
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1
                    
            lists_xy[cluster_index].append(sample[0:2])
            lists_w[cluster_index].append(sample[2])
            
        biggest_shift = 0.0
        for j in range(k):
            #print(len(lists_xy[j]))
            #part_sum = 0
            lists_xy[j] = np.array(lists_xy[j])
            lists_w[j] = np.array(lists_w[j])
            nn = len(lists_xy[j])
            listx_sorted, listw_sorted_1 = w_median(lists_xy[j][:,0], lists_w[j])
            listy_sorted, listw_sorted_2 = w_median(lists_xy[j][:,1], lists_w[j])
            sum_1 = np.sum(listw_sorted_1)
            listw_sorted_11 = listw_sorted_1/sum_1
            sum_2 = np.sum(listw_sorted_2)
            listw_sorted_22 = listw_sorted_2/sum_2
            partw_x_i = part_sum(nn, listw_sorted_11)
            partw_y_i = part_sum(nn, listw_sorted_22)
            #print(listw_sorted_11[:partw_x_i+1])
            if np.sum(listw_sorted_11[:partw_x_i+1]) > 0.50000000000000000 or np.sum(listw_sorted_22[:partw_y_i+1]) > 0.50000000000000000:
                clusters_x = listx_sorted[partw_x_i]
                clusters_y = listy_sorted[partw_y_i]
            elif np.sum(listw_sorted_11[:partw_x_i+1]) == 0.50000000000000000 or np.sum(listw_sorted_22[:partw_y_i+1]) == 0.50000000000000000:
                clusters_x = (listx_sorted[partw_x_i]+listx_sorted[partw_x_i+1])/2
                clusters_y = (listy_sorted[partw_y_i]+listy_sorted[partw_y_i+1])/2
  
            spare_clusters = np.append(spare_clusters, [[clusters_x, clusters_y]], axis=0)
            shift = get_distance(clusters[j], [clusters_x, clusters_y], 1)
            biggest_shift = max(biggest_shift, shift)
        #print(biggest_shift)
        if biggest_shift < cutoff:
            #print("第{}次迭代后，聚类稳定。".format(n_loop))
            break
        else:
            clusters = spare_clusters
            #print(clusters)
            n_loop += 1
            #print(lists_w)
    return clusters, lists_xy, lists_w

#Optimizing the center points with BFGS
def bdfs_l2(samples, k, cutoff):
    #samples_xy = samples[:,0:2].tolist()
    #clusters = random.sample(samples_xy, k)
    #clusters = np.array(clusters)
    clusters = np.array([[0,0],[5,5]])
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        lists_w = [[] for _ in range(k)]
        for sample in samples:
            #print(sample)
            smallest_distance = get_distance(sample[0:2], clusters[0], 2)
            cluster_index = 0
            
            for i in range(k-1):
                distance = get_distance(sample[0:2], clusters[i+1], 2)
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1
                    
            lists_xy[cluster_index].append(sample[0:2])
            lists_w[cluster_index].append(sample[2])
            
        biggest_shift = 0.0
        for j in range(k):
            lists_xy[j] = np.array(lists_xy[j])
            lists_w[j] = np.array(lists_w[j])
            j_clusters = clusters[j]
            j_clusters.resize((1,2))
            #lists_xy[j][:,0].resize((:,1))
            len_lists = len(lists_xy[j])
            w_lists = lists_w[j]
            w_lists.resize((len_lists,1))
            #print(j_clusters)
            #print(lists_xy[j])
            #print(w_lists)
            #result = optim.fmin_l_bfgs_b(weighed_distance, clusters[j], args=(lists_xy[j][:,0], lists_xy[j][:,1], lists_w[j]))
            #init_theta = np.random.normal(size=(1, 2))
            #print(init_theta)
            #dis = weighed_distance(init_theta, lists_xy[j], w_lists)
            #print(lists_xy[j])
            result = optim.fmin_l_bfgs_b(weighed_distance, j_clusters, args=(lists_xy[j], w_lists), approx_grad=True)
            
            spare_clusters = np.append(spare_clusters, [[result[0][0], result[0][1]]], axis=0)
            shift = get_distance(clusters[j], result[0], 2)
            biggest_shift = max(biggest_shift, shift)
            
        if biggest_shift < cutoff:
            #print("第{}次迭代后，聚类稳定。".format(n_loop))
            break
        else:
            clusters = spare_clusters
            #print(clusters)
            n_loop += 1
    return clusters, lists_xy, lists_w

def HL_estimator(criter, array_point):
    n = len(array_point)
    k_hl = []
    if criter == 'hl_1':
        for i in range(n-1):
            for j in range(i+1, n):
                med_value = (array_point[i]+array_point[j])/2
                k_hl.append(med_value)
    elif criter == 'hl_2':
        for i in range(n):
            for j in range(i, n):
                med_value = (array_point[i]+array_point[j])/2
                k_hl.append(med_value)
    elif criter == 'hl_3':
        for i in range(n):
            for j in range(n):
                med_value = (array_point[i]+array_point[j])/2
                k_hl.append(med_value)
                
    k_hl = np.array(k_hl)
    return k_hl


#Optimizing the center points with Manhattan
def hod_mann_hl1(samples, k, cutoff):
    clusters = np.array([[0,0],[5,5]])
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        lists_w = [[] for _ in range(k)]
        for sample in samples:
            smallest_distance = get_distance(sample[0:2], clusters[0], 1)
            cluster_index = 0
            
            for i in range(k-1):
                distance = get_distance(sample[0:2], clusters[i+1], 1)
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1
                    
            lists_xy[cluster_index].append(sample[0:2])
            lists_w[cluster_index].append(sample[2])
            
        biggest_shift = 0.0
        for j in range(k):
            #print(len(lists_xy[j]))
            #part_sum = 0
            lists_xy[j] = np.array(lists_xy[j])
            lists_w[j] = np.array(lists_w[j])
            lists_xhl = HL_estimator('hl_1', lists_xy[j][:,0])
            lists_yhl = HL_estimator('hl_1', lists_xy[j][:,1])
            lists_whl = HL_estimator('hl_1', lists_w[j])
            
            nn = len(lists_whl)
            listx_sorted, listw_sorted_1 = w_median(lists_xhl, lists_whl)
            listy_sorted, listw_sorted_2 = w_median(lists_yhl, lists_whl)
            sum_1 = sum(listw_sorted_1)
            listw_sorted_11 = listw_sorted_1/sum_1
            sum_2 = sum(listw_sorted_2)
            listw_sorted_22 = listw_sorted_2/sum_2
            
            partw_x_i = part_sum(nn, listw_sorted_11)
            partw_y_i = part_sum(nn, listw_sorted_22)
            #print(listw_sorted_11[:partw_x_i+1])
            if sum(listw_sorted_11[:partw_x_i+1]) > 0.50000000000000000:
                clusters_x = listx_sorted[partw_x_i]
            if sum(listw_sorted_22[:partw_y_i+1]) > 0.50000000000000000:
                clusters_y = listy_sorted[partw_y_i]
            elif sum(listw_sorted_11[:partw_x_i+1]) == 0.50000000000000000:
                clusters_x = (listx_sorted[partw_x_i]+listx_sorted[partw_x_i+1])/2
            elif sum(listw_sorted_22[:partw_y_i+1]) == 0.50000000000000000:
                clusters_y = (listy_sorted[partw_y_i]+listy_sorted[partw_y_i+1])/2
  
            spare_clusters = np.append(spare_clusters, [[clusters_x, clusters_y]], axis=0)
            shift = get_distance(clusters[j], [clusters_x, clusters_y], 1)
            biggest_shift = max(biggest_shift, shift)
        #print(biggest_shift)
        if biggest_shift < cutoff:
            #print("第{}次迭代后，聚类稳定。".format(n_loop))
            break
        else:
            clusters = spare_clusters
            #print(clusters)
            n_loop += 1
            #print(lists_w)
    return clusters, lists_xy, lists_w

def hod_mann_hl2(samples, k, cutoff):
    #samples_xy = samples[:,0:2].tolist()
    #clusters = random.sample(samples_xy, k)
    clusters = np.array([[0,0],[5,5]])
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        lists_w = [[] for _ in range(k)]
        for sample in samples:
            smallest_distance = get_distance(sample[0:2], clusters[0], 1)
            cluster_index = 0
            
            for i in range(k-1):
                distance = get_distance(sample[0:2], clusters[i+1], 1)
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1
                    
            lists_xy[cluster_index].append(sample[0:2])
            lists_w[cluster_index].append(sample[2])
            
        biggest_shift = 0.0
        for j in range(k):
            #print(len(lists_xy[j]))
            #part_sum = 0
            lists_xy[j] = np.array(lists_xy[j])
            lists_w[j] = np.array(lists_w[j])
            lists_xhl = HL_estimator('hl_2', lists_xy[j][:,0])
            lists_yhl = HL_estimator('hl_2', lists_xy[j][:,1])
            lists_whl = HL_estimator('hl_2', lists_w[j])
            
            nn = len(lists_whl)
            listx_sorted, listw_sorted_1 = w_median(lists_xhl, lists_whl)
            listy_sorted, listw_sorted_2 = w_median(lists_yhl, lists_whl)
            sum_1 = sum(listw_sorted_1)
            listw_sorted_11 = listw_sorted_1/sum_1
            sum_2 = sum(listw_sorted_2)
            listw_sorted_22 = listw_sorted_2/sum_2
            partw_x_i = part_sum(nn, listw_sorted_11)
            partw_y_i = part_sum(nn, listw_sorted_22)
            #print(listw_sorted_11[:partw_x_i+1])
            if sum(listw_sorted_11[:partw_x_i+1]) > 0.50000000000000000:
                clusters_x = listx_sorted[partw_x_i]
            if sum(listw_sorted_22[:partw_y_i+1]) > 0.50000000000000000:
                clusters_y = listy_sorted[partw_y_i]
            elif sum(listw_sorted_11[:partw_x_i+1]) == 0.50000000000000000:
                clusters_x = (listx_sorted[partw_x_i]+listx_sorted[partw_x_i+1])/2
            elif sum(listw_sorted_22[:partw_y_i+1]) == 0.50000000000000000:
                clusters_y = (listy_sorted[partw_y_i]+listy_sorted[partw_y_i+1])/2
  
            spare_clusters = np.append(spare_clusters, [[clusters_x, clusters_y]], axis=0)
            shift = get_distance(clusters[j], [clusters_x, clusters_y], 1)
            biggest_shift = max(biggest_shift, shift)
        #print(biggest_shift)
        if biggest_shift < cutoff:
            #print("第{}次迭代后，聚类稳定。".format(n_loop))
            break
        else:
            clusters = spare_clusters
            #print(clusters)
            n_loop += 1
            #print(lists_w)
    return clusters, lists_xy, lists_w

def hod_mann_hl3(samples, k, cutoff):
    clusters = np.array([[0,0],[5,5]])
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        lists_w = [[] for _ in range(k)]
        for sample in samples:
            smallest_distance = get_distance(sample[0:2], clusters[0], 1)
            cluster_index = 0
            
            for i in range(k-1):
                distance = get_distance(sample[0:2], clusters[i+1], 1)
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1
                    
            lists_xy[cluster_index].append(sample[0:2])
            lists_w[cluster_index].append(sample[2])
            
        biggest_shift = 0.0
        for j in range(k):
            #print(len(lists_xy[j]))
            #part_sum = 0
            lists_xy[j] = np.array(lists_xy[j])
            lists_w[j] = np.array(lists_w[j])
            lists_xhl = HL_estimator('hl_3', lists_xy[j][:,0])
            lists_yhl = HL_estimator('hl_3', lists_xy[j][:,1])
            lists_whl = HL_estimator('hl_3', lists_w[j])
            
            nn = len(lists_whl)
            listx_sorted, listw_sorted_1 = w_median(lists_xhl, lists_whl)
            listy_sorted, listw_sorted_2 = w_median(lists_yhl, lists_whl)
            sum_1 = sum(listw_sorted_1)
            listw_sorted_11 = listw_sorted_1/sum_1
            sum_2 = sum(listw_sorted_2)
            listw_sorted_22 = listw_sorted_2/sum_2
            
            partw_x_i = part_sum(nn, listw_sorted_11)
            partw_y_i = part_sum(nn, listw_sorted_22)
            #print(listw_sorted_11[:partw_x_i+1])
            if sum(listw_sorted_11[:partw_x_i+1]) > 0.50000000000000000:
                clusters_x = listx_sorted[partw_x_i]
            if sum(listw_sorted_22[:partw_y_i+1]) > 0.50000000000000000:
                clusters_y = listy_sorted[partw_y_i]
            elif sum(listw_sorted_11[:partw_x_i+1]) == 0.50000000000000000:
                clusters_x = (listx_sorted[partw_x_i]+listx_sorted[partw_x_i+1])/2
            elif sum(listw_sorted_22[:partw_y_i+1]) == 0.50000000000000000:
                clusters_y = (listy_sorted[partw_y_i]+listy_sorted[partw_y_i+1])/2
  
            spare_clusters = np.append(spare_clusters, [[clusters_x, clusters_y]], axis=0)
            shift = get_distance(clusters[j], [clusters_x, clusters_y], 1)
            biggest_shift = max(biggest_shift, shift)
        #print(biggest_shift)
        if biggest_shift < cutoff:
            #print("第{}次迭代后，聚类稳定。".format(n_loop))
            break
        else:
            clusters = spare_clusters
            #print(clusters)
            n_loop += 1
            #print(lists_w)
    return clusters, lists_xy, lists_w

def rela_eff(x,a1,b1,a2,b2):
    sum_x11 = 0
    sum_x12 = 0
    sum_x11_x12 = 0
    sum_x21 = 0
    sum_x22 = 0
    sum_x21_x22 = 0
    for i in range(len(x)):
        x11 = x[i][0][0]-a1
        x12 = x[i][0][1]-b1
        sum_x11 += pow(x11,2)
        sum_x12 += pow(x12,2)
        sum_x11_x12 += x11*x12
        
        x21 = x[i][1][0]-a2
        x22 = x[i][1][1]-b2
        sum_x21 += pow(x21,2)
        sum_x22 += pow(x22,2)
        sum_x21_x22 += x21*x22
    return (sum_x11*sum_x12-pow(sum_x11_x12,2))/pow(len(x),2),(sum_x21*sum_x22-pow(sum_x21_x22,2))/pow(len(x),2)

def multivariatet(mu,Sigma,N,M):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Sigma = scale matrix (dxd numpy array)
    N = degrees of freedom
    M = # of samples to produce
    '''
    d = len(Sigma)
    g = np.tile(np.random.gamma(N/2.,2./N,M),(d,1)).T
    Z = np.random.multivariate_normal(np.zeros(d),Sigma,M)
    return mu + Z/np.sqrt(g)


if __name__ == '__main__':

    all_clusters_means = []
    all_clusters_medians = []
    all_clusters_bfgs = []
    all_clusters_hl1 = []
    all_clusters_hl2 = []
    all_clusters_hl3 = []
    
    all_clusters_means_c1 = []
    all_clusters_means_c2 = []
    all_clusters_medians_c1 = []
    all_clusters_medians_c2 = []
    all_clusters_bfgs_c1 = []
    all_clusters_bfgs_c2 = []
    all_clusters_hl1_c1 = []
    all_clusters_hl1_c2 = []
    all_clusters_hl2_c1 = []
    all_clusters_hl2_c2 = []
    all_clusters_hl3_c1 = []
    all_clusters_hl3_c2 = []
    # 随机生成三组二元正态分布随机数
    #np.random.seed(1314)
    for i in range(100):
        np.random.seed(i*3)
        
        """
        #normal
        
        x1, y1 = np.random.normal(0, 1, 50).T,np.random.normal(0, 1, 50).T
        x2, y2 = np.random.normal(5, 1, 50).T,np.random.normal(5, 1, 50).T
        #print(x1,y1)
        """
        """
        #Laplace
        x1, y1 = np.random.laplace(0, 1, 50).T, np.random.laplace(0, 1, 50).T
        x2, y2 = np.random.laplace(5, 1, 50).T, np.random.laplace(5, 1, 50).T
        """
        """
        #Logistic
        x1, y1 = np.random.logistic(0, 1, 50).T, np.random.logistic(0, 1, 50).T
        x2, y2 = np.random.logistic(5, 1, 50).T, np.random.logistic(5, 1, 50).T
        """
        """
        #Uniform
        x1, y1 = np.random.uniform(-1, 1, 50).T, np.random.uniform(-1, 1, 50).T
        x2, y2 = np.random.uniform(4, 6, 50).T, np.random.uniform(4, 6, 50).T
        
        """
        #t
        t_1 = multivariatet([0, 0], [[1, 0], [0, 1]], 110, 50).T
        x1 = t_1[0]
        y1 = t_1[1]
        t_2 = multivariatet([5, 5], [[1, 0], [0, 1]], 110, 50).T
        x2 = t_2[0]
        y2 = t_2[1]
        #print(x1,y1)
        #x_out = np.array([0,3,x2[0],10,30,50,70])
        #y_out = np.array([0,3,y2[0],10,30,50,70])
        #for j,k in zip(x_out, y_out):
        
        # 绘制三组数据的散点图
        x = np.hstack((x1, x2))
        y = np.hstack((y1, y2))
        #print(x,y)
        
        #w_set = creat_weight(len(x))
        #print(sum(w_set))
        w_set = [1/len(x) for i in range(len(x))]
        
        #print(w_set)
        x_y_w = np.column_stack((x,y,w_set))
        #print(x_y_w[:,2])
    
        #print(x_y_w)
        n_clusters_means, coor_xy_means, coor_w_means= kmeans(x_y_w, 2, 0.00000001)
        if n_clusters_means[0][0]>4:
            n_clusters_means[[0,1],:] = n_clusters_means[[1,0],:] #Swap two lines of matrix
        all_clusters_means_c1.append(n_clusters_means[0])
        all_clusters_means_c2.append(n_clusters_means[1])
        
        n_clusters_bfgs, coor_xy_bfgs, coor_w_bfgs= bdfs_l2(x_y_w, 2, 0.00000001)
        if n_clusters_bfgs[0][0]>4:
            n_clusters_bfgs[[0,1],:] = n_clusters_bfgs[[1,0],:] #Swap two lines of matrix
        all_clusters_bfgs_c1.append(n_clusters_bfgs[0])
        all_clusters_bfgs_c2.append(n_clusters_bfgs[1])
            
        
            
        n_clusters_hl2, coor_xy_hl2, coor_w_hl2= hod_mann_hl2(x_y_w, 2, 0.00000001)
        if n_clusters_hl2[0][0]>4:
            n_clusters_hl2[[0,1],:] = n_clusters_hl2[[1,0],:] #Swap two lines of matrix
        all_clusters_hl2_c1.append(n_clusters_hl2[0])
        all_clusters_hl2_c2.append(n_clusters_hl2[1])
            
    
        n_clusters_hl1, coor_xy_hl1, coor_w_hl1= hod_mann_hl1(x_y_w, 2, 0.00000001)
        #print(n_clusters_hl1[0])
        if n_clusters_hl1[0][0]>4:
            n_clusters_hl1[[0,1],:] = n_clusters_hl1[[1,0],:] #Swap two lines of matrix
        all_clusters_hl1_c1.append(n_clusters_hl1[0])
        all_clusters_hl1_c2.append(n_clusters_hl1[1])
        
            
        n_clusters_hl3, coor_xy_hl3, coor_w_hl3= hod_mann_hl3(x_y_w, 2, 0.00000001)
        if n_clusters_hl3[0][0]>4:
            n_clusters_hl3[[0,1],:] = n_clusters_hl3[[1,0],:] #Swap two lines of matrix
        all_clusters_hl3_c1.append(n_clusters_hl3[0])
        all_clusters_hl3_c2.append(n_clusters_hl3[1])
            
        n_clusters_medians, coor_xy_medians, coor_w_medians= manhattan(x_y_w, 2, 0.00000001)
        if n_clusters_medians[0][0]>4:
            n_clusters_medians[[0,1],:] = n_clusters_medians[[1,0],:] #Swap two lines of matrix
        all_clusters_medians_c1.append(n_clusters_medians[0])
        all_clusters_medians_c2.append(n_clusters_medians[1])
    
        #print(coor_xy)
        all_clusters_means.append(n_clusters_means)
        all_clusters_medians.append(n_clusters_medians)
        all_clusters_bfgs.append(n_clusters_bfgs)
        all_clusters_hl1.append(n_clusters_hl1)
        all_clusters_hl2.append(n_clusters_hl2)
        all_clusters_hl3.append(n_clusters_hl3)
        
    #print(all_clusters_medians_c1)
    det1_means,det2_means = rela_eff(all_clusters_means,0,0,5,5)
    print(f'det1_means is {det1_means},',f'det2_means is {det2_means}')
    det_means_sum = det1_means+det2_means
    print(f'det_means_sum is {det_means_sum}')
    r_means = 0.0001082487292447072/det_means_sum
    print(f'r_means is {r_means}')
    
    det1_medians,det2_medians = rela_eff(all_clusters_medians,0,0,5,5)
    print(f'det1_medians is {det1_medians},',f'det2_medians is {det2_medians}')
    det_medians_sum = det1_medians+det2_medians
    print(f'det_medians_sum is {det_medians_sum}')
    r_medians = 0.0001082487292447072/det_medians_sum
    print(f'r_medians is {r_medians}')
    
    det1_bfgs,det2_bfgs = rela_eff(all_clusters_bfgs,0,0,5,5)
    print(f'det1_bfgs is {det1_bfgs},',f'det2_bfgs is {det2_bfgs}')
    det_bfgs_sum = det1_bfgs+det2_bfgs
    print(f'det_bfgs_sum is {det_bfgs_sum}')
    r_bfgs = 0.0001082487292447072/det_bfgs_sum
    print(f'r_bfgs is {r_bfgs}')
    
    det1_hl1,det2_hl1 = rela_eff(all_clusters_hl1,0,0,5,5)
    print(f'det1_hl1 is {det1_hl1},',f'det2_hl1 is {det2_hl1}')
    det_hl1_sum = det1_hl1+det2_hl1
    print(f'det_hl1_sum is {det_hl1_sum}')
    r_hl1 =0.0001082487292447072/det_hl1_sum
    print(f'r_hl1 is {r_hl1}')
    
    det1_hl2,det2_hl2 = rela_eff(all_clusters_hl2,0,0,5,5)
    print(f'det1_hl2 is {det1_hl2},',f'det2_hl2 is {det2_hl2}')
    det_hl2_sum = det1_hl2+det2_hl2
    print(f'det_hl2_sum is {det_hl2_sum}')
    r_hl2 = 0.0001082487292447072/det_hl2_sum
    print(f'r_hl2 is {r_hl2}')
    
    det1_hl3,det2_hl3 = rela_eff(all_clusters_hl3,0,0,5,5)
    print(f'det1_hl3 is {det1_hl3},',f'det2_hl3 is {det2_hl3}')
    det_hl3_sum = det1_hl3+det2_hl3
    print(f'det_hl3_sum is {det_hl3_sum}')
    r_hl3 = 0.0001082487292447072/det_hl3_sum
    print(f'r_hl3 is {r_hl3}')
    
    plt.subplot()
    plt.grid(True,ls="-.")
    #plt.axis('tight')
    #plt.xlabel(r'$x_{1}$',fontproperties='SimHei',fontsize=12)             #设置x，y轴的标签
    plt.xlabel('X',fontproperties='SimHei',fontsize=12)
    plt.ylabel('Y',fontproperties='SimHei',fontsize=12)
    plt.text(x=-0.5,#文本x轴坐标 
             y=5.5, #文本y轴坐标
             s='Method A', #文本内容
             
             fontdict=dict(fontsize=15, color='black',family='serif',),#字体属性字典
             
             #添加文字背景色
             bbox={#'facecolor': '#74C476', #填充色
                  'edgecolor':'b',#外框色
                   'alpha': 0, #框透明度
                   'pad': 8,#本文与框周围距离 
                  })

    for k, samples in enumerate(all_clusters_means):
        x = []
        y = []
        # random.choice
        # color = [color_names[i]] * len(c.samples)
        for sample in samples:
            x.append(sample[0])
            y.append(sample[1])
        plt.scatter(x, y)
    #plt.plot(n_clusters[:,0], n_clusters[:,1], 'r^')
    #plt.plot(j, k, 'ko')
    plt.show()
    
    plt.grid(True,ls="-.")
    plt.xlabel('X',fontproperties='SimHei',fontsize=12)
    plt.ylabel('Y',fontproperties='SimHei',fontsize=12)
    plt.text(x=-0.5,#文本x轴坐标 
             y=5.5, #文本y轴坐标
             s='Method B', #文本内容
             
             fontdict=dict(fontsize=15, color='black',family='serif',),#字体属性字典
             
             #添加文字背景色
             bbox={#'facecolor': '#74C476', #填充色
                  'edgecolor':'b',#外框色
                   'alpha': 0, #框透明度
                   'pad': 8,#本文与框周围距离 
                  })

    for k, samples in enumerate(all_clusters_medians):
        x = []
        y = []
        # random.choice
        # color = [color_names[i]] * len(c.samples)
        for sample in samples:
            x.append(sample[0])
            y.append(sample[1])
        plt.scatter(x, y)
    plt.show()
    
    plt.grid(True,ls="-.")
    plt.xlabel('X',fontproperties='SimHei',fontsize=12)
    plt.ylabel('Y',fontproperties='SimHei',fontsize=12)
    plt.text(x=-0.5,#文本x轴坐标 
             y=5.5, #文本y轴坐标
             s='Method C', #文本内容
             
             fontdict=dict(fontsize=15, color='black',family='serif',),#字体属性字典
             
             #添加文字背景色
             bbox={#'facecolor': '#74C476', #填充色
                  'edgecolor':'b',#外框色
                   'alpha': 0, #框透明度
                   'pad': 8,#本文与框周围距离 
                  })

    for k, samples in enumerate(all_clusters_bfgs):
        x = []
        y = []
        # random.choice
        # color = [color_names[i]] * len(c.samples)
        for sample in samples:
            x.append(sample[0])
            y.append(sample[1])
        plt.scatter(x, y)
    plt.show()
    
    plt.grid(True,ls="-.")
    plt.xlabel('X',fontproperties='SimHei',fontsize=12)
    plt.ylabel('Y',fontproperties='SimHei',fontsize=12)
    plt.text(x=-0.5,#文本x轴坐标 
             y=5.5, #文本y轴坐标
             s='Method D', #文本内容
             
             fontdict=dict(fontsize=15, color='black',family='serif',),#字体属性字典
             
             #添加文字背景色
             bbox={#'facecolor': '#74C476', #填充色
                  'edgecolor':'b',#外框色
                   'alpha': 0, #框透明度
                   'pad': 8,#本文与框周围距离 
                  })

    for k, samples in enumerate(all_clusters_hl1):
        x = []
        y = []
        # random.choice
        # color = [color_names[i]] * len(c.samples)
        for sample in samples:
            x.append(sample[0])
            y.append(sample[1])
        plt.scatter(x, y)
    plt.show()
    
    plt.grid(True,ls="-.")
    plt.xlabel('X',fontproperties='SimHei',fontsize=12)
    plt.ylabel('Y',fontproperties='SimHei',fontsize=12)
    plt.text(x=-0.5,#文本x轴坐标 
             y=5.5, #文本y轴坐标
             s='Method E', #文本内容
             
             fontdict=dict(fontsize=15, color='black',family='serif',),#字体属性字典
             
             #添加文字背景色
             bbox={#'facecolor': '#74C476', #填充色
                  'edgecolor':'b',#外框色
                   'alpha': 0, #框透明度
                   'pad': 8,#本文与框周围距离 
                  })

    for k, samples in enumerate(all_clusters_hl2):
        x = []
        y = []
        # random.choice
        # color = [color_names[i]] * len(c.samples)
        for sample in samples:
            x.append(sample[0])
            y.append(sample[1])
        plt.scatter(x, y)
    plt.show()
    
    plt.grid(True,ls="-.")
    plt.xlabel('X',fontproperties='SimHei',fontsize=12)
    plt.ylabel('Y',fontproperties='SimHei',fontsize=12)
    plt.text(x=-0.5,#文本x轴坐标 
             y=5.5, #文本y轴坐标
             s='Method F', #文本内容
             
             fontdict=dict(fontsize=15, color='black',family='serif',),#字体属性字典
             
             #添加文字背景色
             bbox={#'facecolor': '#74C476', #填充色
                  'edgecolor':'b',#外框色
                   'alpha': 0, #框透明度
                   'pad': 8,#本文与框周围距离 
                  })

    for k, samples in enumerate(all_clusters_hl3):
        x = []
        y = []
        # random.choice
        # color = [color_names[i]] * len(c.samples)
        for sample in samples:
            x.append(sample[0])
            y.append(sample[1])
        plt.scatter(x, y)
    plt.show()
    
    x_means_c1 = []
    y_means_c1 = []
    x_medians_c1 = []
    y_medians_c1 = []
    x_bfgs_c1 = []
    y_bfgs_c1 = []
    x_hl1_c1 = []
    y_hl1_c1 = []
    x_hl2_c1 = []
    y_hl2_c1 = []
    x_hl3_c1 = []
    y_hl3_c1 = []
    #plt.grid(True,ls="-.")
    plt.style.use('ggplot')
    plt.xlabel('X',fontproperties='SimHei',fontsize=12)
    plt.ylabel('Y',fontproperties='SimHei',fontsize=12)
    plt.text(x=0.61,#文本x轴坐标 
         y=-1.42, #文本y轴坐标
         s='(0,0)', #文本内容
         
         fontdict=dict(fontsize=13, color='black',family='serif',),#字体属性字典
         
         #添加文字背景色
         bbox={#'facecolor': '#74C476', #填充色
              'edgecolor':'b',#外框色
               'alpha': 0, #框透明度
               'pad': 8,#本文与框周围距离 
              })
    
    font1 = {'size':8,}

    for k_means_c1, sample_means_c1 in enumerate(all_clusters_means_c1):
        x_means_c1.append(sample_means_c1[0])
        y_means_c1.append(sample_means_c1[1])
    plt.scatter(x_means_c1, y_means_c1,s=6,marker='^',label='Method A')
    
    for k_medians_c1, sample_medians_c1 in enumerate(all_clusters_medians_c1):
        x_medians_c1.append(sample_medians_c1[0])
        y_medians_c1.append(sample_medians_c1[1])
    plt.scatter(x_medians_c1, y_medians_c1,s=6,marker='o',label='Method B')
    
    for k_bfgs_c1, sample_bfgs_c1 in enumerate(all_clusters_bfgs_c1):
        x_bfgs_c1.append(sample_bfgs_c1[0])
        y_bfgs_c1.append(sample_bfgs_c1[1])
    plt.scatter(x_bfgs_c1, y_bfgs_c1,s=6,marker='s',label='Method C')
    
    for k_hl1_c1, sample_hl1_c1 in enumerate(all_clusters_hl1_c1):
        x_hl1_c1.append(sample_hl1_c1[0])
        y_hl1_c1.append(sample_hl1_c1[1])
    plt.scatter(x_hl1_c1, y_hl1_c1,s=6,marker='s',label='Method $D_{1}$')
    
    for k_hl2_c1, sample_hl2_c1 in enumerate(all_clusters_hl2_c1):
        x_hl2_c1.append(sample_hl2_c1[0])
        y_hl2_c1.append(sample_hl2_c1[1])
    plt.scatter(x_hl2_c1, y_hl2_c1,s=6,marker='8',label='Method $D_{2}$')
    
    for k_hl3_c1, sample_hl3_c1 in enumerate(all_clusters_hl3_c1):
        x_hl3_c1.append(sample_hl3_c1[0])
        y_hl3_c1.append(sample_hl3_c1[1])
    plt.scatter(x_hl3_c1, y_hl3_c1,s=6,marker='D',label='Method $D_{3}$')
    plt.legend(loc='upper left',prop=font1)
    plt.xlim(-1.2,1.0)
    plt.ylim(-1.8,1.8)
    plt.show()
    
    x_means_c2 = []
    y_means_c2 = []
    x_medians_c2 = []
    y_medians_c2 = []
    x_bfgs_c2 = []
    y_bfgs_c2 = []
    x_hl1_c2 = []
    y_hl1_c2 = []
    x_hl2_c2 = []
    y_hl2_c2 = []
    x_hl3_c2 = []
    y_hl3_c2 = []
    plt.xlabel('X',fontproperties='SimHei',fontsize=12)
    plt.ylabel('Y',fontproperties='SimHei',fontsize=12)
    plt.text(x=5.62,#文本x轴坐标 
         y=4.3, #文本y轴坐标
         s='(5,5)', #文本内容
         
         fontdict=dict(fontsize=13, color='black',family='serif',),#字体属性字典
         
         #添加文字背景色
         bbox={#'facecolor': '#74C476', #填充色
              'edgecolor':'b',#外框色
               'alpha': 0, #框透明度
               'pad': 8,#本文与框周围距离 
              })
    for k_means_c2, sample_means_c2 in enumerate(all_clusters_means_c2):
        x_means_c2.append(sample_means_c2[0])
        y_means_c2.append(sample_means_c2[1])
    plt.scatter(x_means_c2, y_means_c2,s=6,marker='^',label='Method A')
    
    for k_medians_c2, sample_medians_c2 in enumerate(all_clusters_medians_c2):
        x_medians_c2.append(sample_medians_c2[0])
        y_medians_c2.append(sample_medians_c2[1])
    plt.scatter(x_medians_c2, y_medians_c2,s=6,marker='o',label='Method B')
    
    for k_bfgs_c2, sample_bfgs_c2 in enumerate(all_clusters_bfgs_c2):
        x_bfgs_c2.append(sample_bfgs_c2[0])
        y_bfgs_c2.append(sample_bfgs_c2[1])
    plt.scatter(x_bfgs_c2, y_bfgs_c2,s=6,marker='s',label='Method C')
    
    for k_hl1_c2, sample_hl1_c2 in enumerate(all_clusters_hl1_c2):
        x_hl1_c2.append(sample_hl1_c2[0])
        y_hl1_c2.append(sample_hl1_c2[1])
    plt.scatter(x_hl1_c2, y_hl1_c2,s=6,marker='s',label='Method $D_{1}$')
    
    for k_hl2_c2, sample_hl2_c2 in enumerate(all_clusters_hl2_c2):
        x_hl2_c2.append(sample_hl2_c2[0])
        y_hl2_c2.append(sample_hl2_c2[1])
    plt.scatter(x_hl2_c2, y_hl2_c2,s=6,marker='v',label='Method $D_{2}$')
    
    for k_hl3_c2, sample_hl3_c2 in enumerate(all_clusters_hl3_c2):
        x_hl3_c2.append(sample_hl3_c2[0])
        y_hl3_c2.append(sample_hl3_c2[1])
    plt.scatter(x_hl3_c2, y_hl3_c2,s=6,marker='D',label='Method $D_{3}$')
    plt.legend(loc='upper left',prop=font1)
    plt.xlim(3.8,6.0)
    plt.ylim(4.0,7.0)
    plt.show()

    
#plt.grid(axis="y")