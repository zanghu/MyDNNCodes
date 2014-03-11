#coding: utf8
from new_numpy_GBRBM import MyGaussianBinaryRBM
#from pylearn2.datasets.preprocessing import GlobalContrastNormalization, ZCA, Pipeline, ExtractPatches
#from module.preprocessing import PCA, GlobalContrastNormalization, ZCA, Pipeline, ExtractPatches
import numpy
import time
import cPickle

#1.获得训练样本集的设计矩阵
f = open('/home/zanghu/pca69_mnist_pylearn2.pkl') 
dsm = cPickle.load(f)
f.close()
data = dsm.X
assert data.shape == (50000, 69)
#data = numpy.load('HL_mnist_train.npy')

#2.模型对象的创建与训练
rbm_model = MyGaussianBinaryRBM(n_vis=69, n_hid=200)

t0 = time.clock()
rbm_model.train(data=data, lr=0.001, batch_size=20, epochs=15, k=1, momentum=0.0)
t1 = time.clock()
print 'time elapsed on training is', t1 - t0

#3.保存模型
import cPickle
f = open('HL_mnist_model.pkl', 'w')
cPickle.dump(rbm_model,  f, protocol=2)
f.close()
