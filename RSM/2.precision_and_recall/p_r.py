# -*- coding: utf-8 -*-
import numpy
import scipy
import cPickle
import numpy
from collections import OrderedDict
import time
import pylab

#train_label_path, test_label_path: str(字符串)，储存训练样本类标的文本路径与储存测试样本类标的文本路径.
#由于计算机中存放数据集的路径是不会经常变化的，所以这两个参数没有作为函数参数而是作为全局变量，需要使用者根据自己的情况手动设置
orin_train_label_path = '../orin_data/train.label'
orin_test_label_path = '../orin_data/test.label'
train_data_path = '../data/20news_train.npy'       
test_data_path = '../data/20news_test.npy'
train_label_path = '../data/20news_train_label.npy'
test_label_path = '../data/20news_test_label.npy'
        

def rsm_query(model_path, prefix='', suffix='', save_results=False, pic=True):
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
    #f = open('/home/zanghu/实验save/rsm/20news_train.npy')
    f = open(train_data_path)
    X_train = numpy.load(f)
    f.close()
        
    #f = open('/home/zanghu/实验save/rsm/20news_test.npy')
    f = open(test_data_path)
    X_test = numpy.load(f)
    f.close()
        
    #f = open('/home/zanghu/实验save/rsm/20news_train_label.npy')
    f = open(train_label_path)
    y_train = numpy.load(f)
    f.close()
        
    #f = open('/home/zanghu/实验save/rsm/20news_test_label.npy')
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
    
    #2.求平均精度和平均召回率
    #2.1文档向量归一化，为了下一步计算余弦距离
    #X_train = X_train[:600, :] #测试程序用的数据子集
    #X_test = X_test[:600, :] #测试程序用的数据子集
    train_normalizer = numpy.sqrt(numpy.sum(X_train**2, axis=1))
    #若投影后为零向量，则范数设为1，在做除法时， 0/1仍等于0
    for n, i in enumerate(train_normalizer):
        if i == 0.:
            train_normalizer[n] = 1.
    assert numpy.all(train_normalizer) # 检查分母不为零
    X_train = X_train / train_normalizer[:, numpy.newaxis]
    test_normalizer = numpy.sqrt(numpy.sum(X_test**2, axis=1))
    for n, i in enumerate(test_normalizer):
        if i == 0.:
            test_normalizer[n] = 1.
    assert all(test_normalizer) # 检查分母不为零
    X_test = X_test / test_normalizer[:, numpy.newaxis]
        
    distance_matrix = numpy.dot(X_test, X_train.T) # 乘积得到的矩阵的行标代表测试集样本，列标代表训练集样本
    t1 = time.clock()
    print '预处理时间:', t1 - t0
    
    #2.2.计算测试集上每个query的精度和召回率, 使用隐主题进行数据挖掘，每个测试集样本作为一个query
    precision_matrix = numpy.zeros(shape=distance_matrix.shape) # 精度记录矩阵
    recall_matrix = numpy.zeros(shape=distance_matrix.shape) # 召回率记录矩阵
    dict_list = []
    for cnt, row in enumerate(distance_matrix):
        row_dict = {}
        for k, num in enumerate(row):
            row_dict[k] = num # row_dict字典中的键是训练集文档的索引号，值是该训练集文档和该行对应的测试集文档之间的余弦距离
        od = OrderedDict(sorted(row_dict.items(), key=lambda t:t[1], reverse=True)) # 依据余弦相似度从大到小排列
        p = []; r = []
        current_recall = 0. # 当前召回数
        #current_precision = 0. # 当前精度
        total = train_cnt[y_test[cnt]] # 测试集中索引cnt的文档的类标是y_test[cnt]，该类标在样本集中的总数为total
        for i, key in enumerate(od.keys()):
            if(y_train[key] == y_test[cnt]):
                current_recall = current_recall + 1.
            p.append(current_recall / (i + 1.)) # 记录精度
            r.append(current_recall / total) # 记录召回率
    
        precision_matrix[cnt, :] = p #记录索引号cnt的query的精度
        recall_matrix[cnt, :] = r #记录索引号cnt的query的召回率
        
    #2.3.求测试集所有query在训练集上的平均召回率和平均精度
    mean_precision = numpy.mean(precision_matrix, axis=0)
    mean_recall = numpy.mean(recall_matrix, axis=0)
    t2 = time.clock()
    print '主机算过程用时:', t2 - t1
    #print 'time elapsed:', t2 - t1
    
    #3.保存计算结果
    if save_results:
        numpy.save('precision_matrix.npy', precision_matrix)
        numpy.save('recall_matrix.npy', recall_matrix)
        numpy.save('mean_precision.npy', mean_precision)
        numpy.save('mean_recall.npy', mean_recall)
    print 'model saved...'
    
    f = open(prefix+'mean_precision_record'+suffix, 'w')
    for num, i in enumerate(mean_precision):
        f.write(str(num) + ': ' + str(i) + '\n')
    f.close()
    
    f = open(prefix+'mean_recall_record'+suffix, 'w')
    for num, i in enumerate(mean_recall):
        f.write(str(num) + ': ' + str(i) + '\n')
    f.close()
    if pic is True:
        print 'drawing curve...'
        pr_curve(mean_precision=mean_precision, mean_recall=mean_recall, prefix=prefix, suffix=suffix) #绘制召回率-精度曲线
    print 'all finished...'

#绘制p-r曲线
def pr_curve(mean_precision, mean_recall, prefix='', suffix=''):
    """
    Parameters
    ----------

    mean_precision: numpy.ndarry，一维
        记录平均精度的向量.
    mean_recall: numpy.ndarray，一维
        记录平均召回率向量.
    prefix, duffix: 同rsm_query()
    """
     
    #color_list = ['blue', 'red', 'green', 'cyan', 'yellow', 'black', 'magenta', (0.5,0.5,0.5)]
    y_vector = mean_precision * 100.
    x_vector = mean_recall #** (1./6.)

    pylab.figure(figsize=(8, 8))
    pylab.grid() #在做标系中显示网格
    pylab.plot(x_vector, y_vector, label='$r-p curve$', color='blue', linewidth=1)

    pylab.xlabel('recall(%)')
    pylab.ylabel('precision(%)')
    pylab.title('20-newsgroups')
    #pylab.xlim(0., 60) #x轴长度限制
    pylab.ylim(0., 60) #y轴长度限制
    pylab.legend() #在图像中显示标记说明
    #pylab.show() # 显示图像
    pylab.savefig(prefix + 'r-p.png' + suffix, dpi=320) #保存图像，可以人为指定所保存的图像的分辨率
    print 'pic saved...'
    
    
