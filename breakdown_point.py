# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 21:34:31 2020

@author: ASL
"""
import math
import matplotlib.pyplot as plt

def f_mean(n):
    return int(1)/n

def f_median(n):
    return int((n+1)/2)/n

def f_hl1(n):
    a = n+0.5-math.sqrt((n-0.5)**2-2*int((n**2-n-2)/4))
    return int(a)/n

def f_hl2(n):
    b = n+1.5-math.sqrt((n+0.5)**2-2*int((n**2+n-2)/4))
    return int(b)/n
   
def f_hl3(n):
    c = n+1-math.sqrt(n**2-int((n**2-1)/2))
    return int(c)/n

if __name__ == '__main__':
    mean_values = []
    median_values = []
    hl1_values = []
    hl2_values = []
    hl3_values = []
    for i in range(1,51):
        a1 = f_mean(i)
        b1 = f_median(i)
        c1 = f_hl1(i)
        d1 = f_hl2(i)
        e1 = f_hl3(i)
        mean_values.append(a1)
        median_values.append(b1)
        hl1_values.append(c1)
        hl2_values.append(d1)
        hl3_values.append(e1)
    print(hl3_values[8],hl1_values[8])
    plt.xlabel('n',fontproperties='SimHei',fontsize=12)
    plt.ylabel('Finite-sample replacement breakdown point',fontproperties='SimHei',fontsize=12)
    plt.plot(mean_values,'-',label='Mean',color='saddlebrown')
    plt.plot(median_values,label='Median',color='darkgoldenrod')
    plt.plot(median_values,label='$L^{2}$-median',color='darkgoldenrod')
    plt.plot(hl1_values,label='HL1',color='b',linewidth=2,linestyle='--')
    plt.plot(hl2_values,label='HL2',color='g',linewidth=2,linestyle=':')
    
    plt.plot(hl3_values,'.',label='HL3',color='red')
    plt.grid(True)
    plt.legend(loc = 0)
    plt.show()