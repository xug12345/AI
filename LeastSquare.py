# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 18:04:26 2018

@author: XUGANG
"""

#encoding=UTF-8  
''''' 
Created on 2014年6月30日 
 
@author: jin 
'''  
import numpy as np
import matplotlib.pyplot as plt  
#import random 
  
def loadData():  
    x = np.arange(-1,1,0.02)  
    
    y = ((x*x-1)**3+1)*(np.cos(x*2)+0.6*np.sin(x*1.3))  
    #生成的曲线上的各个点偏移一下，并放入到xa,ya中去  
    xr=[];yr=[];i = 0  
    for xx in x:  
        yy=y[i]  
#        d=float(random.randint(80,120))/100  
        i+=1  
        xr.append(xx)  
        yr.append(yy)    
    return x,y,xr,yr  
def XY(x,y,order):  
    X=[]  
    for i in range(order+1):  
        X.append(x**i)  
    X=np.mat(X).T  
    Y=np.array(y).reshape((len(y),1))  
    return X,Y  
def figPlot(x1,y1,x2,y2):  
    plt.plot(x1,y1,color='g',linestyle='-',marker='')  
    plt.plot(x2,y2,color='m',linestyle='',marker='.')  
    plt.show()  
def Main(order):      
    x,y,xr,yr = loadData()  
#    print(x,y,xr,yr)
    X,Y = XY(x,y,order)  
    XT=X.transpose()#X的转置  
#    inverse=np.linalg.inv(np.dot(XT,X))
#    B=np.dot(inverse,Y)
    B=np.dot(np.dot(np.linalg.inv(np.dot(XT,X)),XT),Y)#套用最小二乘法公式  
    myY=np.dot(X,B)  
    figPlot(x,myY,xr,yr)  
    delta=myY-Y
    delta=np.transpose(delta)*delta
    
    print('params size of',B.size,'delta=',delta)
    print(B)
    
Main(15)  