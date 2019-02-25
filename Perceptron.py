# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:28:45 2018

@author: XUGANG
"""
import torch as t

class Perceptron(object):
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        '''
        self.activator = activator
        # 权重向量初始化为0
        self.weights = t.rand(input_num)
        self.bias=t.rand(1);
        # 偏置项初始化为0
    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)
    
    def predict(self, x):
        '''
        输入向量，输出感知器的计算结果
        '''
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        y=x*self.weights
        yy=y.sum()+self.bias
        yyy=self.activator(yy)

        return yyy
    def train(self, input_vecs, labels, iteration, rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
            
    def _one_iteration(self, input_vecs, labels, rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        for i in range(input_vecs.size(0)):
            output = self.predict(input_vecs[i])
            # 更新权重
            self._update_weights(input_vecs[i], output, labels[i], rate)
            
    def _update_weights(self, x, y, label, rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - y
        delta *= rate
        
        self.weights= self.weights+delta * x
        # 更新bias
        self.bias += delta
        
def f(x):
     '''
     定义激活函数f    
     '''
     if(x>0):
         return 1
     else:
         return 0
 #    t.round(t.sigmoid(x))
 #    return t.round(t.sigmoid(x))
        
def get_training_dataset():
     '''
     基于and真值表构建训练数据
     '''
     # 构建训练数据
     # 输入向量列表
     input_vecs =t.Tensor([[0,0] ,[0,1],[1,0], [1,1]])
     # 期望的输出列表，注意要与输入一一对应
     # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
     labels = t.Tensor([1, 1, 1, 0])
     return input_vecs, labels    

def train_nand_perceptron():
     '''
     使用and真值表训练感知器
     '''
     # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
     p = Perceptron(2, f)
     # 训练，迭代10轮, 学习速率为0.1
     input_vecs, labels = get_training_dataset()
     p.train(input_vecs, labels, 10, 0.1)
     #返回训练好的感知器
     return p

def xor(nand,a,b):
     x=nand.predict(t.Tensor([a,b]))
     y=nand.predict(t.Tensor([x,b]))
     z=nand.predict(t.Tensor([x,a]))
     w=nand.predict(t.Tensor([y,z]))
     return w

def nand(p,a,b):
     return p.predict(t.Tensor([a,b]))

if __name__ == '__main__': 
     # 训练and感知器
     p = train_nand_perceptron()
     # 打印训练获得的权重
     print(p)
     # 测试
     print('0 nand 0 = %d' % nand(p,0,0))
     print('0 nand 1 = %d' % nand(p,0,1))
     print('1 nand 0 = %d' % nand(p,1,0))
     print('1 nand 1 = %d' % nand(p,1,1))
     print('0 xor 0 = %d' % xor(p,0,0))
     print('0 xor 1 = %d' % xor(p,0,1))
     print('1 xor 0 = %d' % xor(p,1,0))
     print('1 xor 1 = %d' % xor(p,1,1))
