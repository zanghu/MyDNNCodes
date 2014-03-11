#coding=utf-8
from HonglakLees_CRBM import MyConvRBM
import cPickle
from pylearn2.datasets.mnist import MNIST
import numpy
import theano
import theano.tensor as T
import pylab

f = open('/home/zanghu/mm_conv/conv_rbm/my_crbm_trainsave.pkl')
crbm_model = cPickle.load(f)
f.close()

theano.config.floatX = 'float32'
filters, c, b = crbm_model.get_params()
print type(filters)
print type(c)
print type(b)
#print dir(filters.__class__.__name__)
dsm = MNIST(which_set='train', start=0, stop=100)
for i in xrange(10):
    for j in xrange(10):
        pylab.subplot(10, 10, i*10+j+1); pylab.axis('off'); pylab.imshow(dsm.X[i*10+j].reshape(28, 28), cmap=pylab.cm.gray)
pylab.savefig('/home/zanghu/mm_conv/conv_rbm/orin.png', dpi=240)
#pylab.imshow(dsm.X[0].reshape(28, 28), cmap=pylab.cm.gray)
#pylab.show()
x = numpy.cast['float32'](dsm.X[:100, :].reshape(100, 1, 28, 28))

y = T.tensor4()
f = theano.function([y], crbm_model.gibbs_vhv(y)[-1])
#[h0_mean, h0_sample, pool0_mean, pool0_sample, v1_mean, v1_sample] = crbm_model.gibbs_vhv(x)
for k in xrange(100):
    x = f(x)
    for i in xrange(10):
        for j in xrange(10):
            pylab.subplot(10, 10, i*10+j+1); pylab.axis('off'); pylab.imshow(x[i*10+j].reshape(28, 28), cmap=pylab.cm.gray)
    pylab.savefig('/home/zanghu/mm_conv/conv_rbm/plots/conv_recon_'+'epoch_'+str(k+1)+'.png', dpi=240)
pylab.show()
