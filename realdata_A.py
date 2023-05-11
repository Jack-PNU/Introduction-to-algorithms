# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:14:33 2020

@author: OS
"""

#from __future__ import print_function

import numpy as np
import random
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer
import scipy.optimize as optim
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

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

def nw_distance(theta, x_y):
    #for i in range(len(x_y)):
        #dist = np.sum(w[i] * np.linalg.norm(x_y[i]-theta[i], ord=k))
    #x_y = np.column_stack((x,y))
    num = x_y.shape[0]
    dist = 0.0
    for i in range(num):
        distmat = np.linalg.norm(x_y[i]-theta, ord=2)
        #distmat = w[i]*distmat
        dist = dist + distmat
    return dist

def sum_distance(theta, x_y):
    #for i in range(len(x_y)):
        #dist = np.sum(w[i] * np.linalg.norm(x_y[i]-theta[i], ord=k))
    #x_y = np.column_stack((x,y))
    num = x_y.shape[0]
    dist = 0.0
    for i in range(num):
        distmat = np.linalg.norm(x_y[i]-theta[i], ord=2)
        #distmat = w[i]*distmat
        dist = dist + distmat
    return dist

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
def kmeans(samples, clusters, k, cutoff):
    #samples_xy = samples[:,0:2].tolist()
    #clusters = random.sample(samples_xy, k)
    #clusters = np.array(clusters)
    #clusters = np.array([[-3,0],[0,-0.5],[3,0.5]])
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        for sample in samples:
            smallest_distance = get_distance(sample[0:2], clusters[0],2)
            cluster_index = 0
            
            for i in range(k-1):
                distance = get_distance(sample[0:2], clusters[i+1],2)
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1
                    
            lists_xy[cluster_index].append(sample[0:2])
        biggest_shift = 0.0
        for j in range(k):
            #print(len(lists_xy[j]))
            lists_xy[j] = np.array(lists_xy[j])
            clusters_x = np.mean(lists_xy[j][:,0])
            clusters_y = np.mean(lists_xy[j][:,1])
            

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
    
    return clusters, lists_xy

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
def kmedians(samples, clusters, k, cutoff): 
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        for sample in samples:
            smallest_distance = get_distance(sample[0:2], clusters[0],1)
            cluster_index = 0
            
            for i in range(k-1):
                distance = get_distance(sample[0:2], clusters[i+1],1)
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1
                    
            lists_xy[cluster_index].append(sample[0:2])
        biggest_shift = 0.0
        for j in range(k):
            #print(len(lists_xy[j]))
            lists_xy[j] = np.array(lists_xy[j])
            lists_xhl = lists_xy[j][:,0]
            lists_yhl = lists_xy[j][:,1]
            clusters_x = np.median(lists_xhl)
            clusters_y = np.median(lists_yhl)
            

            spare_clusters = np.append(spare_clusters, [[clusters_x, clusters_y]], axis=0)
            shift = get_distance(clusters[j], [clusters_x, clusters_y],1)
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
    
    return clusters, lists_xy

def kbdfs_l2(samples, clusters, k, cutoff):
    #samples_xy = samples[:,0:2].tolist()
    #clusters = random.sample(samples_xy, k)
    #clusters = np.array(clusters)
    #clusters = np.array([[-3,0],[0,-0.5],[3,0.5]])
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
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
            
        biggest_shift = 0.0
        for j in range(k):
            lists_xy[j] = np.array(lists_xy[j])
            j_clusters = clusters[j]
            j_clusters.resize((1,2))
            #lists_xy[j][:,0].resize((:,1))
            #print(j_clusters)
            #print(lists_xy[j])
            #print(w_lists)
            #result = optim.fmin_l_bfgs_b(weighed_distance, clusters[j], args=(lists_xy[j][:,0], lists_xy[j][:,1], lists_w[j]))
            #init_theta = np.random.normal(size=(1, 2))
            #print(init_theta)
            #dis = weighed_distance(init_theta, lists_xy[j], w_lists)
            #print(lists_xy[j])
            result = optim.fmin_l_bfgs_b(nw_distance, j_clusters, args=(lists_xy[j],), approx_grad=True)
            
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
    return clusters, lists_xy

#Optimizing the center points with BFGS
def bdfs_l2(samples, clusters, k, cutoff):
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

def k_hl(samples, clusters, k, cutoff, hl):

    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        for sample in samples:
            smallest_distance = get_distance(sample[0:2], clusters[0], 2)
            cluster_index = 0
            
            for i in range(k-1):
                distance = get_distance(sample[0:2], clusters[i+1], 2)
                if distance < smallest_distance:
                    smallest_distance = distance
                    cluster_index = i + 1
                    
            lists_xy[cluster_index].append(sample[0:2])
            
        biggest_shift = 0.0
        for j in range(k):
            #print(len(lists_xy[j]))
            #part_sum = 0
            lists_xy[j] = np.array(lists_xy[j])
            lists_xhl = HL_estimator(hl, lists_xy[j][:,0])
            lists_yhl = HL_estimator(hl, lists_xy[j][:,1])
            clusters_x = np.median(lists_xhl)
            clusters_y = np.median(lists_yhl)
  
            spare_clusters = np.append(spare_clusters, [[clusters_x, clusters_y]], axis=0)
            shift = get_distance(clusters[j], [clusters_x, clusters_y], 2)
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
    return clusters, lists_xy

def hod_mann_hl(samples, clusters, k, cutoff, hl):
    #samples_xy = samples[:,0:2].tolist()
    #clusters = random.sample(samples_xy, k)
    #clusters = np.array(clusters)
    #clusters = np.array([[-3,0],[0,-0.5],[3,0.5]])
    n_loop = 0
    while True:
        #print(clusters)
        spare_clusters = np.empty(shape=[0, 2])
        lists_xy = [[] for _ in range(k)]
        lists_w = [[] for _ in range(k)]
        for sample in samples:
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
            #print(len(lists_xy[j]))
            #part_sum = 0
            lists_xy[j] = np.array(lists_xy[j])
            lists_w[j] = np.array(lists_w[j])
            lists_xhl = HL_estimator(hl, lists_xy[j][:,0])
            lists_yhl = HL_estimator(hl, lists_xy[j][:,1])
            lists_whl = HL_estimator(hl, lists_w[j])
            
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
            shift = get_distance(clusters[j], [clusters_x, clusters_y], 2)
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

def re_assign(samples, clusters, metric):
    k = len(clusters)
    lists_xy = [[] for _ in range(k)]
    for sample in samples:
        smallest_distance = get_distance(sample[0:2], clusters[0], metric)
        cluster_index = 0
        for i in range(k-1):
            distance = get_distance(sample[0:2], clusters[i+1], metric)
            if distance < smallest_distance:
                smallest_distance = distance
                cluster_index = i + 1
        lists_xy[cluster_index].append(sample[0:2])
    return lists_xy

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
    
    mean_distance = []
    median_distance = []
    l2_median_distance = []
    hl_distance = []
    
    for i in range(10000):
        
        iris = load_iris()
        data = iris.data
        
        """
        wine = pd.read_csv('wine-clustering.csv')
        data = np.array(wine)
        """
        
        """
        seed = pd.read_csv('Seed_Data.csv')
        data = seed.iloc[:, 0:7]
        data = np.array(data)
        """
        data_copy = data.copy()
        
        pca = PCA(n_components=2)#实例化
        pca = pca.fit(data)#拟合模型
        data = pca.transform(data)#获取新矩阵

        """
        num = np.random.choice(len(data), 5, replace=False)
        #print(num)
        for j in range(5):
            data[num[j]]= data[num[j]]*2
        """
        #noise = np.random.normal(0, 0.1, len(data))
        noise = [np.sin(j+i*0.4)*0.2 for j in range(len(data))]
        for i in range(len(data)):
            #data[i] = data[i] + np.random.randn()-0.5
            data[i] = data[i] + noise[i]
        
        
        pca = PCA(n_components=2)#实例化
        pca = pca.fit(data_copy)#拟合模型
        data_copy = pca.transform(data_copy)#获取新矩阵

        w_set = [1 for i in range(len(data))]
        
        x_y_w = np.column_stack((data,w_set))
        x_y_w_copy = np.column_stack((data_copy,w_set))
        
        k=3
        
        row_rand_array = np.arange(data.shape[0])
        np.random.shuffle(row_rand_array)
        clusters = data[row_rand_array[0:k]]
        
        clusters = np.array([[-3,0],[0,-0.5],[3,0.5]])
        #clusters = np.array([[-400,0],[0,0],[600,0]])
        #clusters = np.array([[-4,1],[0,-1],[4,0]])
        
        no_conta = []
        conta = []
        
        n_clusters_means, coor_xy_means = kmeans(x_y_w, clusters, k, 0.00000001)
        no_conta.append(n_clusters_means)
        #print(n_clusters_means)
        n_clusters_means_copy, _ = kmeans(x_y_w_copy, clusters, k, 0.00000001)
        conta.append(n_clusters_means_copy)
        
        dist_mean = sum_distance(n_clusters_means, n_clusters_means_copy)
        mean_distance.append(dist_mean)

        
        n_clusters_medians, coor_xy_medians = kmedians(x_y_w, clusters, k, 0.00000001)
        no_conta.append(n_clusters_medians)
        #print(n_clusters_medians)
        n_clusters_medians_copy, _ = kmedians(x_y_w_copy, clusters, k, 0.00000001)
        conta.append(n_clusters_medians_copy)
        
        dist_median = sum_distance(n_clusters_medians, n_clusters_medians_copy)
        median_distance.append(dist_median)

        n_clusters_bfgs, coor_xy_bfgs = kbdfs_l2(x_y_w, clusters, k, 0.00000001)
        no_conta.append(n_clusters_bfgs)
        #print(n_clusters_bfgs)
        n_clusters_bfgs_copy, _ = kbdfs_l2(x_y_w_copy, clusters, k, 0.00000001)
        conta.append(n_clusters_bfgs_copy)
        
        dist_l2_median = sum_distance(n_clusters_bfgs, n_clusters_bfgs_copy)
        l2_median_distance.append(dist_l2_median)
        #n_clusters_bfgs_, coor_xy_bfgs_, coor_w_bfgs= bdfs_l2(x_y_w, clusters, 3, 0.00000001)
        #print(n_clusters_bfgs_)

        #n_clusters_hl1, coor_xy_hl1, coor_w_hl1= hod_mann_hl1(x_y_w, clusters, k, 0.00000001)
        #print(n_clusters_hl1)

        #n_clusters_hl2, coor_xy_hl2, coor_w_hl2= hod_mann_hl2(x_y_w, clusters, k, 0.00000001)
        #print(n_clusters_hl2)

        
        n_clusters_hl1, coor_xy_hl1 = k_hl(data, clusters, k, 0.00000001, 'hl_1')
        no_conta.append(n_clusters_hl1)
        #print(n_clusters_hl1)
        n_clusters_hl1_copy, _ = k_hl(x_y_w_copy, clusters, k, 0.00000001, 'hl_1')
        conta.append(n_clusters_hl1_copy)
        
        dist_hl = sum_distance(n_clusters_hl1, n_clusters_hl1_copy)
        hl_distance.append(dist_hl)
        
        n_clusters_hl2, coor_xy_hl2 = k_hl(data, clusters, k, 0.00000001, 'hl_2')
        #print(n_clusters_hl2)
        
        n_clusters_hl3, coor_xy_hl3 = k_hl(data, clusters, k, 0.00000001, 'hl_3')
        #print(n_clusters_hl3)

        all_clusters_means.append(n_clusters_means)
        all_clusters_medians.append(n_clusters_medians)
        all_clusters_bfgs.append(n_clusters_bfgs)
        all_clusters_hl1.append(n_clusters_hl1)

    
    print(np.mean(mean_distance))
    print(np.mean(median_distance))
    print(np.mean(l2_median_distance))
    print(np.mean(hl_distance))

    coor_means = re_assign(data_copy, n_clusters_means,2)
    print(len(coor_means[0]),len(coor_means[1]),len(coor_means[2]))
    coor_medians = re_assign(data_copy, n_clusters_medians,1)
    print(len(coor_medians[0]),len(coor_medians[1]),len(coor_medians[2]))
    coor_bfgs = re_assign(data_copy, n_clusters_bfgs,2)
    print(len(coor_bfgs[0]),len(coor_bfgs[1]),len(coor_bfgs[2]))
    #coor_bfgs_ = re_assign(data_copy, n_clusters_bfgs_,2)
    #print(len(coor_bfgs_[0]),len(coor_bfgs_[1]),len(coor_bfgs_[2]))
    coor_hl1 = re_assign(data_copy, n_clusters_hl1,2)
    print(len(coor_hl1[0]),len(coor_hl1[1]),len(coor_hl1[2]))
    coor_hl2 = re_assign(data_copy, n_clusters_hl2,2)
    coor_hl3 = re_assign(data_copy, n_clusters_hl3,2)
    
    #画出分类图
    plt.style.use('ggplot')
    color = ["red","blue","orange"]
    plt.xlabel('PCA1',fontproperties='SimHei',fontsize=10)
    plt.ylabel('PCA2',fontproperties='SimHei',fontsize=10)
    for i in [0,1,2]:
        coor_means[i] = np.array(coor_means[i])
        plt.scatter(coor_means[i][:,0],
                    coor_means[i][:,1],
                    s=10,
                    c=color[i],
                    alpha = 0.5)#透明度
    plt.xlim(-3.5,4)
    plt.show()
    #plt.scatter(coor_xy_means[0][:,0],coor_xy_means[0][:,1],c="red")
    #plt.scatter(coor_xy_means[1][:,0],coor_xy_means[1][:,1],c = "black")
    #plt.scatter(coor_xy_means[2][:,0],coor_xy_means[2][:,1],c="orange")
    #plt.legend()#显示图例
    #plt.title("PCA of IRIS dataset")#显示标题
    
    plt.xlabel('PCA1',fontproperties='SimHei',fontsize=10)
    plt.ylabel('PCA2',fontproperties='SimHei',fontsize=10)
    for i in [0,1,2]:
        coor_medians[i] = np.array(coor_medians[i])
        plt.scatter(coor_medians[i][:,0],
                    coor_medians[i][:,1],
                    c=color[i],
                    s=10,
                    alpha = 0.5)#透明度
    plt.xlim(-3.5,4)
    plt.show()
    
    plt.xlabel('PCA1',fontproperties='SimHei',fontsize=10)
    plt.ylabel('PCA2',fontproperties='SimHei',fontsize=10)
    for i in [0,1,2]:
        coor_bfgs[i] = np.array(coor_bfgs[i])
        plt.scatter(coor_bfgs[i][:,0],
                    coor_bfgs[i][:,1],
                    c=color[i],
                    s=10,
                    alpha = 0.5)#透明度
    plt.xlim(-3.5,4)
    plt.show()
    
    plt.xlabel('PCA1',fontproperties='SimHei',fontsize=10)
    plt.ylabel('PCA2',fontproperties='SimHei',fontsize=10)
    for i in [0,1,2]:
        coor_hl1[i] = np.array(coor_hl1[i])
        plt.scatter(coor_hl1[i][:,0],
                    coor_hl1[i][:,1],
                    c=color[i],
                    s=10,
                    alpha = 0.5)#透明度
    plt.xlim(-3.5,4)
    plt.show()
    
    for i in [0,1,2]:
        coor_hl2[i] = np.array(coor_hl2[i])
        plt.scatter(coor_hl2[i][:,0],
                    coor_hl2[i][:,1],
                    c=color[i],
                    s=20,
                    alpha = 0.7)#透明度
    #plt.xlim(-3.5,4)
    plt.show()
    
    for i in [0,1,2]:
        coor_hl3[i] = np.array(coor_hl3[i])
        plt.scatter(coor_hl3[i][:,0],
                    coor_hl3[i][:,1],
                    c=color[i],
                    s=20,
                    alpha = 0.7)#透明度
    #plt.xlim(-3.5,4)
    plt.show()