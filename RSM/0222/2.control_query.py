# -*- coding: utf-8 -*-
import numpy
import scipy
import cPickle
import numpy
from collections import OrderedDict
import time

from new_query_nade import p_r

#train_label_path, test_label_path: str(字符串)，储存训练样本类标的文本路径与储存测试样本类标的文本路径.
#由于计算机中存放数据集的路径是不会经常变化的，所以这两个参数没有作为函数参数而是作为全局变量，需要使用者根据自己的情况手动设置
orin_train_label_path = 'train.label'
orin_test_label_path = 'test.label'
train_data_path = '20news_train.npy'       
test_data_path = '20news_test.npy'
train_label_path = '20news_train_label.npy'
test_label_path = '20news_test_label.npy'
        

def rsm_query(model_path, prefix='', suffix='', save_results=False, pic=True, show_error=False):
    """
    计算rsm模型的召回率矩阵，精度矩阵，召回率向量，精度向量

    Parameters
    ----------

    model_path: str(字符串)
        训练好的模型的完整路径或相对路径，比如'/home/yanyan/.../rsm.pkl'.
    prefix: str(字符串)，optional
        保存精度和召回率的文件的名字的前缀.
    suffix: str(字符串)，optional
        保存精度和召回率的文件的名字的后缀.
    save_results: bool(布尔型), optional
        是否保存计算得到的精度矩阵，召回率矩阵，平均召回率向量，平均精度向量.
    pic: bool(布尔型), optional
        是否绘制"平均召回率-平均精度"曲线.
    """ 
    #0.声明全局变量
    global orin_train_label_path
    global orin_test_label_path
    global train_data_path    
    global test_data_path
    global train_label_path
    global test_label_path
    #1.准备工作，读取相关信息
    #1.1.读取类标信息，分别记录训练集和测试集中每个类别有多少个样本
    #label_train = []
    f = open(orin_train_label_path)
    cnt = numpy.zeros(20) #20news数据集只有20个类
    for line in f:
        l = line.strip('\n').split(' ')
        #label_train.append(eval(l[0]) - 1) #注意文档类标从1开始，而程序中类标从0开始
        cnt[eval(l[0])-1] += 1
    
    train_cnt = numpy.asarray(cnt, dtype='int64')
    #label_train = numpy.asarray(label_train, dtype='float64')
    print '训练集每一类中样本数:', train_cnt
    print '训练集样本总数:', numpy.sum(cnt)
    
    #label_test = []
    f = open(orin_test_label_path)
    cnt = numpy.zeros(20)
    for line in f:
        l = line.strip('\n').split(' ')
        #label_test.append(eval(l[0]) - 1) # 注意文档类标从1开始，而程序中类标从0开始
        cnt[eval(l[0])-1] += 1
    
    test_cnt = numpy.asarray(cnt, dtype='int64')
    #label_test = numpy.asarray(label_train, dtype='float64')
    print '测试集每一类中样本数:', test_cnt
    print '测试集样本总数:', numpy.sum(cnt)

    #1.2.读取训练好的模型，作为从2000维输入空间到50维主题空间的映射函数
    f = open(model_path)
    rsm = cPickle.load(f)
    f.close()
    
    W = rsm['w_vh']
    b = rsm['w_v']
    c = rsm['w_h']
    
    #1.3.读取训练集样本，测试集样本和他们各自的类标
    f = open(train_data_path)
    X_train = numpy.load(f)
    f.close()
        
    f = open(test_data_path)
    X_test = numpy.load(f)
    f.close()
        
    f = open(train_label_path)
    y_train = numpy.load(f)
    f.close()
        
    f = open(test_label_path)
    y_test = numpy.load(f)
    f.close()
    
    def sigmoid(X):
        return (1. + scipy.tanh(X/2.))/2.
    
    #1.4.将训练样本和测试样本分别投影到主题空间中
    t0 = time.clock()
    v = X_train # 训练集在主题空间的投影
    D = numpy.sum(a=v, axis=1) # 由于使用词袋模型，因此每个文档的词向量的所有分量之和等于该文档的总词数
    sigmoid_activation = numpy.dot(v, W)+ numpy.outer(D, c)
    h_mean = sigmoid(sigmoid_activation)
    X_train = h_mean

    v = X_test # 测试集在主题空间中的投影
    D = numpy.sum(a=v, axis=1) # 由于使用词袋模型，因此每个文档的词向量的所有分量之和等于该文档的总词数
    sigmoid_activation = numpy.dot(v, W)+ numpy.outer(D, c)
    h_mean = sigmoid(sigmoid_activation)
    X_test = h_mean
        
    print 'affine finish...'
    print '投影数据时间: ', time.clock() - t0
    
#调整参数========================================
#X_train代表作为答案的设计矩阵，label_train为对应的类标向量
#X_train代表作为query的设计矩阵，label_train为对应的类标向量
#鉴于如果使用全部的训练集或测试集时，输出文件query_error.txt太大（3-10G)，可以考虑只使用一个子集
#比如下面的参数中，query矩阵和类标向量都进行了切片，取前测试集的前100个样本作为query，这样做的话，输出文件大小大约50MB
    p_r(X_train=X_test, label_train=y_test, X_test=X_test[:100, :], label_test=y_test[:100], pic=True, prefix=prefix, suffix=suffix, save_results=save_results, show_error=show_error)

if __name__ == '__main__':

    path = 'rsm_result_batch001.pkl'

    print 'computing...'
    rsm_query(model_path=path, show_error=True)
