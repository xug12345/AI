# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:39:14 2018

@author: XUGANG
"""

import torch as t
#%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display

# 设置随机数种子，保证在不同电脑上运行时下面的输出一致
t.manual_seed(1000) 

def get_fake_data(batch_size=8):
    ''' 产生随机数据：y=x*2+3，加上了一些噪声'''
    x = t.rand(batch_size, 1) * 20
    y = x * 2 + (1 + t.randn(batch_size, 1))*3
    return x, y

m=30
x,y=get_fake_data(m)
plt.scatter(x, y)

w = t.rand(1, 2) 

lr =0.001 # 学习率

#for ii in range(20000):
#    x, y = get_fake_data()
#    
#    # forward：计算loss
#    y_pred = x.mm(w) + b.expand_as(y) # x@W等价于x.mm(w);for python3 only
#    loss = 0.5 * (y_pred - y) ** 2 # 均方误差
#    loss = loss.sum()
#    
#    # backward：手动计算梯度
#    dloss = 1
#    dy_pred = dloss * (y_pred - y)
#    
#    dw = x.t().mm(dy_pred)
#    db = dy_pred.sum()
#    
#    # 更新参数
#    w-=lr * dw
#    b-=lr * db
    
for ii in range(20000):
    x, y = get_fake_data(m)
    xx=t.cat((x,t.ones(x.size())), 1)
    yy=(xx*w).sum(1).view(y.size())
    
    loss=(yy-y)**2
    loss=loss.sum()/m/2
    
    dloss=(yy-y)*xx
    dloss=dloss.sum(0)/m
    
#    dw=((yy-y)*x).sum()/m
#    db=(yy-y).sum()/m
#    loss=loss.t().mm(loss)
#    loss*=0.5
    
    w-=dloss*lr
    
    if(ii%1000==0):
        print('w=',w,'loss=',loss)
        display.clear_output(wait=True)
        x = t.arange(0, 20).view(-1, 1)
        xx=t.cat((x,t.ones(x.size())), 1)
        y = xx*w
        y=y.sum(1)
        plt.plot(x.numpy(), y.numpy()) # predicted
        
        x2, y2 = get_fake_data(batch_size=20) 
        plt.scatter(x2.numpy(), y2.numpy()) # true data
        
        plt.xlim(0, 20)
        plt.ylim(0, 41)
        plt.show()
    
