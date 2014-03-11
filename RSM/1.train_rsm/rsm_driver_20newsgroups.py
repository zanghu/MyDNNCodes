# -*- coding: utf-8 -*-
"""
Train RSM on 20 Newsgroups dataset (wordcounts)
"""
import rsm_numpy_pro_cd, time, cPickle
import numpy
import gzip
import tarfile
#from pylearn2.datasets.mnist import MNIST

# hyperparameters
hiddens = 50
batchsize = 100
epochs = 1000
learning_rate = 0.001

f = open('../data/20news_train.npy')
#f.close()
#fh = cpickle.load(f)
#data = numpy.load('20news_train.npy')
data = numpy.load(f)
f.close()
# train RSM
start_time = time.time()
r = data.shape[0]/batchsize*batchsize
#layer = rsm_numpy.RSM()
#layer = rsm_numpy_pro_cd.RSM()
layer = rsm_numpy_pro_cd.RSM()
#result = layer.train(data[0:r], units=hiddens, epochs=epochs, lr=learning_rate, btsz=batchsize)
#注意re_typr参数使用来选择训练时重构误差的计算方法，可以选择'per_batch'或者'per_epoch'
#不关选择那种方式，对最终训练结果都没有影响，为了和原作者代码统一，建议选择'per_batch'
result = layer.train(data[0:r], units=hiddens, epochs=epochs, lr=learning_rate, k=1, btsz=batchsize, re_type='per_batch')
print "Time: " + str(time.time() - start_time)

# save results
fh = open("rsm_result_batch1.pkl", "w")
cPickle.dump(result, fh)
fh.close()
