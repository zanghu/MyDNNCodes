# -*- coding: utf-8 -*-
"""
@author:  Joerg Landthaler
@credits: Christian Osendorfer
@date:    Nov, 2011
@organization: TUM, I6, Machine Learning Group
@summary: Implementation of the Replicated Softmax model,
          as presented by R. Salakhutdinov & G.E. Hinton
          in http://www.mit.edu/~rsalakhu/papers/repsoft.pdf
@version: 1.0
@
"""
import scipy as sp
scipy = sp
import numpy

class RSM(object):
    """
    RSM模型
    1.可以选择显示的重构误差的类型
    2.支持cd-k算法中k值的设置
    3.增强了代码的模块化
    """
    def __init__(self, rng=None, seed=19920120):
        """构造函数"""
        if rng is None:
            rng = numpy.random.RandomState(seed=seed)
        self.rng = rng 

    def train(self, data, units, k=1, epochs=1000, lr=0.001, weightinit=0.001, 
            momentum=0.9, btsz=100, re_type='per_batch'):
        """
        训练函数
        Parameters
        ----------

        data: numpy.ndarray, 2维
            训练数据集，采用设计矩阵格式.
        units: int, 网络的隐单元数.
        epochs: int, 训练轮数.
        lr: float，学习速率.
        weightinit: scaling of random weight initialization
        momentum: float
            动量参数，若不希望使用动量法则该值设置为零.
        btsz: int, batchsize.   
        """
        print
        print "[RSM_NUMPY] Training with CD-1, hidden:", units

        self.data = data
        self.k = k # cd-k
        self.lr = lr
        self.momentum = float(momentum)
        self.n_hid = units 
        dictsize = data.shape[1]
        self.n_vis = dictsize
        self.btsz = btsz
        # initilize weights
        self.w_vh = weightinit * numpy.random.randn(dictsize, units)
        self.w_v = weightinit * numpy.random.randn(dictsize)
        self.w_h = weightinit * numpy.random.randn(units)
        # weight updates
        self.wu_vh = numpy.zeros((dictsize, units))
        self.wu_v = numpy.zeros((dictsize))
        self.wu_h = numpy.zeros((units))
        self.epochs = epochs
        self.batches = data.shape[0]/self.btsz
        
        if re_type is 'per_batch': 
            err_list = self.cd_k_updates(epochs=self.epochs, k=1)
        if re_type is 'per_epoch':
            err_list = self.cd_k_updates_1(epochs=self.epochs, k=1)

        return {"w_vh": self.w_vh, 
                "w_v": self.w_v, 
                "w_h": self.w_h,
                "err": err_list}

    def cd_k_updates(self, epochs, k=1): # 在每次cd过程中，计算当前batch的recon_error

        delta = self.lr / self.btsz
        print 'recon_per_batch'
        print "learning_rate: %f" % self.lr
        print "updates per epoch: %s | total updates: %s"%(self.batches, self.batches*self.epochs)
        err_list = []
        for epoch in xrange(epochs):
            print "[RSM_NUMPY] Epoch", epoch
            err = []
            for b in xrange(self.batches):
                start = b * self.btsz 
                v1 = self.data[start: start+self.btsz] # 本次迭代所用的batch对应的样本v1
                # cd-k approximation
                [h1_mean, h1_sample, v2_mean, v2_sample, h2_mean, h2_sample] = self.cd_k_alg(k=self.k, v1=v1)

                h1 = h1_mean #h1_sample 
                v2 = v2_sample
                h2 = h2_mean #h2_sample
                # compute updates
                self.wu_vh = self.wu_vh * self.momentum + numpy.dot(v1.T, h1) - numpy.dot(v2.T, h2)
                self.wu_v = self.wu_v * self.momentum + v1.sum(axis=0) - v2.sum(axis=0)
                self.wu_h = self.wu_h * self.momentum + h1.sum(axis=0) - h2.sum(axis=0)
                # update 
                self.w_vh += self.wu_vh * delta 
                self.w_v += self.wu_v * delta
                self.w_h += self.wu_h * delta
                # calculate reconstruction error
                err.append(((v2_sample-v1)**2).mean())
            mean = numpy.mean(err)
            print "Mean squared error: " + str(mean)
            err_list.append(float(mean))
        return err_list

    def cd_k_updates_1(self, epochs, k=1): # 每个epoch之后，使用一次抽样计算recon_error

        print 'recon_per_epoch'
        delta = self.lr / self.btsz
        print "learning_rate: %f" % self.lr
        print "updates per epoch: %s | total updates: %s"%(self.batches, self.batches*self.epochs)
        err_list = []
        for epoch in xrange(epochs):
            print "[RSM_NUMPY] Epoch", epoch
            for b in xrange(self.batches):
                start = b * self.btsz 
                v1 = self.data[start: start+self.btsz] # 本次迭代所用的batch对应的样本v1
                # cd-k approximation
                [h1_mean, h1_sample, v2_mean, v2_sample, h2_mean, h2_sample] = self.cd_k_alg(k=self.k, v1=v1)

                h1 = h1_mean 
                v2 = v2_sample
                h2 = h2_mean
                # compute updates
                self.wu_vh = self.wu_vh * self.momentum + numpy.dot(v1.T, h1) - numpy.dot(v2.T, h2)
                self.wu_v = self.wu_v * self.momentum + v1.sum(axis=0) - v2.sum(axis=0)
                self.wu_h = self.wu_h * self.momentum + h1.sum(axis=0) - h2.sum(axis=0)
                # update 
                self.w_vh += self.wu_vh * delta 
                self.w_v += self.wu_v * delta
                self.w_h += self.wu_h * delta
            #compute recon_error
            err = []
            for b in xrange(self.batches):
                start = b * self.btsz 
                v1 = self.data[start: start+self.btsz] # 本次迭代所用的batch对应的样本v1
                # cd-k approximation
                v2_mean, v2 = self.cd_k_alg(k=1, v1=v1, reconstruction=True)
                err.append(((v2_mean - v1) ** 2).mean())
            mean = numpy.mean(err)
            print "Mean squared error: " + str(mean)
            err_list.append(float(mean))
        return err_list
        
    # cd-k算法的核心抽样部分，其实就是一次交替Gibbs抽样，该方法被cd_k_updates()和cd_k_updates_1()调用
    def cd_k_alg(self, k, v1, reconstruction=False): 
        D = numpy.sum(v1, axis=1)
        h1_act = numpy.dot(v1, self.w_vh) + numpy.outer(D, self.w_h)
        h1_mean = sigmoid(h1_act)
        h1 = self.rng.binomial(n=1, p=h1_mean)

        #chain_start = h1
        v_list = []; v_mean_list = []
        h_list = []; h_mean_list = []
        h_list.append(h1); h_mean_list.append(h1_mean)
        for i in xrange(k):
            v2_act = numpy.dot(h1, self.w_vh.T) + self.w_v
            numerator = numpy.exp(v2_act)
            denominator = numpy.sum(numerator, axis=1).reshape(v1.shape[0], 1)
            # 注意区分v2_mean和v2_pdf的区别
            v2_pdf = numerator / denominator
            v2_mean = D[:, numpy.newaxis] * v2_pdf
            v2 = numpy.zeros(shape=v2_mean.shape)
            for j in xrange(self.btsz):
                v2[j, :] = self.rng.multinomial(n=D[j], pvals=v2_pdf[j, :], size=None)
            #如果是一次重构误差检验
            if reconstruction == True:
                return v2_mean, v2
        
            h2_act = numpy.dot(v2, self.w_vh) + numpy.outer(D, self.w_h)
            h2_mean = sigmoid(h2_act)
            h2_rand = self.rng.rand(self.btsz, self.n_hid) # h_rand是一个batch_size行，"隐单元个数"个列的矩阵，矩阵每个元素都是依0-1之间均匀分布抽样得到
            h2 = numpy.array(h2_rand < h2_mean, dtype=int)
            #h2 = self.rng.binomial(n=1, p=h2_mean) #使用numpy自带的伯努力抽样方法也可以实现抽样过程，但是效率比较低，所以不推荐

            v_list.append(v2); v_mean_list.append(v2_mean)
            h_list.append(h2); h_mean_list.append(h2_mean)

            h1 = h2
        return h_mean_list[0], h_list[0], v_mean_list[-1], v_list[-1], h_mean_list[-1], h_list[-1]
        

def sigmoid(X):
    """
    sigmoid of X
    """
    return (1 + sp.tanh(X/2))/2
