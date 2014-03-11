#coding=utf-8
import time
import os
from itertools import izip
import copy

import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compat.python2x import OrderedDict

from pylearn2.base import Block
from pylearn2.models.model import Model
from pylearn2.linear.linear_transform import LinearTransform
from pylearn2.costs.cost import Cost, SumOfCosts
from pylearn2.models.mlp import MLP, Layer
from pylearn2.datasets.mnist import MNIST
from pylearn2.space import VectorSpace, Conv2DSpace
from pylearn2.linear.conv2d import make_random_conv2D, make_sparse_random_conv2D
#from conv2d import Conv2D, make_random_conv2D, make_sparse_random_conv2D
from pylearn2.utils import sharedX, safe_union

from module.dataset_from_design import DatasetFromDesign



theano.config.compute_test_value = 'off'

class HonglakLeeSparse(Cost):
    
    def __init__(self, p=0.02):
        self.p = p
    
    def expr(self, model, data):
        
        v = data
        p_h_given_v_matrix = model.propup(v)[-1]
        sum_meta = T.square(self.p - T.mean(p_h_given_v_matrix, axis=0, dtype=theano.config.floatX))
        expr = T.sum(sum_meta)
        
        return expr
    
    def get_data_specs(self, model):
        return (model.get_input_space(), model.get_input_source())

class MyContrastiveDivergence(Cost):
    
    def __init__(self, k, chain_num=None): # CD-k
        # k: CD-k
        self.k = k
        self.chain_num = chain_num
	
    def expr(self, model, data):
        
	    return None
        
    def get_data_specs(self, model):
        return model.get_monitoring_data_specs()

class MyCD_energy_scan(MyContrastiveDivergence):
    
    def get_gradients(self, model, data, ** kwargs):
        #print 'get_gradients'
        pos_v = data
        v_samples = pos_v
        [h_mean, h_samples, pool_mean, pool_samples, vis_mean, vis_samples], scan_updates = theano.scan(fn = model.gibbs_vhv, sequences=None, 
		                        outputs_info=[None, None, None, None, None, v_samples], non_sequences=None, n_steps=self.k)
        pos_h = h_mean[0]
        neg_v = vis_samples[-1]
        neg_h = model.sample_hp_given_v(v=neg_v, sample=False)[0]
        
        cost = -(- model.energy(pos_v, pos_h).mean() + model.energy(neg_v, neg_h).mean())

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs = 'ignore', consider_constant=[pos_v, pos_h, neg_v, neg_h])

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()
        
        updates.update(scan_updates) # add scan_updates

        return gradients, updates

class MyConvRBM(Model, Block):
    """Convolutional Restricted Boltzmann Machine (CRBM)  """
    def __init__(self, input_space, output_channels, pool_shape, batch_size=None, detector_axes=('b', 'c', 0, 1), 
                    kernel_shape=(2,2), kernel_stride=(1, 1), border_mode='valid', 
                    transformer=None, h_bias=None, v_bias=None, numpy_rng=None,theano_rng=None):
        """
        vis_space: Conv2DSpace
        transformer: pylearn2.linear.Conv2D instance
        h_bias: vector, 大小等于输出的feature maps数，每个分量对应一个feature map
        v_bias: vector, 大小等于输入的feature maps数，每个分量对应一个feature map
        pool_shape:
        pool_stride: 根据Honglak Lee的原文，pool区域无交叠，于是要求pool_stride=pool_shape，因此暂时不单独设置pool_stride参数
        需要注意，对于卷积RBM，其隐层对应于卷积后的detector_layer，而输出则对应与pool_layer，因此相对于普通RBM只有输入和输出两个space，CRBM有三个space
        """
        Model.__init__(self) # self.names_to_del = set(); self._test_batch_size = 2
        Block.__init__(self) # self.fn = None; self.cpu_only = False
        
        self.kernel_shape = kernel_shape
        self.kernel_stride = kernel_stride
        self.pool_shape = pool_shape
        self.pool_stride = pool_shape
        self.border_mode = border_mode
        
        self.batch_size = batch_size
        self.force_batch_size = batch_size
        
        input_shape = input_space.shape
        input_channels = input_space.num_channels
        if self.border_mode == 'valid':
            detector_shape = [(input_shape[0] - kernel_shape[0])/int(kernel_stride[0]) + 1, (input_shape[1] - kernel_shape[1])/kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            detector_shape = [(input_shape[0] + kernel_shape[0])/int(kernel_stride[0]) - 1, (input_shape[1] + kernel_shape[1])/kernel_stride[1] - 1]
        
        assert isinstance(input_space, Conv2DSpace)
        self.input_space = input_space # add input_space
        self.detector_space = Conv2DSpace(shape=detector_shape, num_channels=output_channels, axes=detector_axes) # add detector_space
        
        #当前只考虑detector layer的feature map可以被pool_shape无交叠完整分割的情况
        #今后需要补充：边缘补齐的情况
        output_shape = (detector_shape[0] / pool_shape[0], detector_shape[1] / pool_shape[1])
        self.output_space = Conv2DSpace(shape=output_shape, num_channels=output_channels, axes=detector_axes) # add output_space

        self.n_vis = numpy.prod(input_space.shape) * input_space.num_channels
        self.n_hid = detector_shape[0] * detector_shape[1] * output_channels

        if numpy_rng is None:
            # create a number generator
            numpy_rng = numpy.random.RandomState(seed=19900418)
        self.numpy_rng = numpy_rng

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.theano_rng = theano_rng

        if transformer is None:
            irange = 4 * numpy.sqrt(6. / (self.n_hid + self.n_vis))
            transformer = make_random_conv2D(irange=irange, input_space=self.input_space, output_space=self.detector_space, 
                                             kernel_shape = self.kernel_shape, batch_size = self.batch_size, 
                                             subsample = kernel_stride,border_mode = self.border_mode, rng=self.numpy_rng)
        else:
            assert isinstance(transformer, Conv2D)

        if h_bias is None:
            # create shared variable for hidden units bias
            h_bias = theano.shared(value=numpy.zeros(self.detector_space.num_channels, dtype=theano.config.floatX), name='h_bias', borrow=True)

        if v_bias is None:
            # create shared variable for visible units bias
            v_bias = theano.shared(value=numpy.zeros(self.input_space.num_channels, dtype=theano.config.floatX), name='v_bias', borrow=True)

        self.transformer = transformer
        self.h_bias = h_bias
        self.v_bias = v_bias

        self._params = safe_union(self.transformer.get_params(), [self.h_bias, self.v_bias])
        
    def get_monitoring_data_specs(self):
        """
        这里虽然仍然调用get_input_space()方法，但由于self.input_space已经是一个Conv2DSpace实例，因此dataset再进行迭代循环时产生的不是design_mat而是topo_view
        """
        return (self.get_input_space(), self.get_input_source())
		
    def get_monitoring_channels(self, data):
        """"""
        v = data
        channels = {}
        
		# recon_error
        channel_name = 'recon_error'
        p_v_given_h, v_sample = self.gibbs_vhv(v)[4:]
        recon_error = ((p_v_given_h - v) ** 2).mean(axis=0).sum()
        channels[channel_name] = recon_error
        
        return channels
    
    # CRBM类应该提供卷积核集合W
    # W： 4Dtensor
    # self.transformer是一个Conv2D实例
    def energy(self, v, h):
        # h: 4D tensor, 轴向('b', 'c', 0, 1)
        # v: 4D tensor, 轴向('b', 'c', 0, 1)
    
        # 利用Conv2D实例的lmul方法，实现正向卷积过程计算，计算结果conv_feature_maps是全体detector_layer的feature map
        conv_feature_maps = self.transformer.lmul(v)
        # 要得导能量函数，将4D tensor h与conv_feature-maps对应样本的对应feature map(channel)的对应位置元素相乘
        vWh = (h * conv_feature_maps).sum(axis=[1,2,3])
        # 求能量函数中与偏置有关的项
        ch = T.sum(self.h_bias * T.sum(h, axis=[2,3]), axis=1)
        bv = T.sum(self.v_bias * T.sum(v, axis=[2,3]), axis=1)
        # E是一个向量，其每个分量对应于该位置样本的能量
        E = - vWh - ch - bv
    
        return E
    
    #模块1: 计算卷积，即计算detector_layer收到的输入信号
    def get_activation(self, v):
        """
        v: 4D-tensor
        本方法的意义在于，p(h|v)是一个sigmoid函数，对于隐单元h_k_mn，其激活函数为activation的第k个feature map的索引为(m, n)单元的值
        """
        conv_activation = self.transformer.lmul(v) # activation也是4D-tensor
        activation = conv_activation + self.h_bias.dimshuffle(0, 'x', 'x') # 加入偏置
        return activation
    
    #模块2 + 模块3：计算p(h|v), p_pool和抽样h_sample和pool_sample
    def sample_hp_given_v(self, v, sample=False):
        """
        本方法的目的，是获得h和pool的概率密度“矩阵“，事实上这两个"矩阵"都是4D-tensor

        本方法的理论，首先透过现象看本质，即虽然能量函数涉及了卷积的计算，导致能量函数的形式非常复杂，但事实上只是将一维的隐“层”和显“层”reshape成了三维的张量
        但能量函数本身诱导的条件概率的表达式（sigmoid函数）和独立性（不同隐单元h_i_mn条件独立）都没有改变，因此在没有pool层的情况下，理论分析推导是简单的
        当考虑pool层时，等于对detector_layer加入了约束，要求处在同一个pool区块中的隐单元至多有一个处于激活态，经过理论推导，发现计算这种约束下的概率密度，
        需要首先计算出detector_layer的卷积输入值

        在编程方面，本方法也涉及大量技巧，包括：
        exp(x)函数的安全计算方法
        平面区域上多个区块的提高效率的pool方法（平移切片）
        高维概率密度张量转化为矩阵来形式，实现多项式抽样
        高维张量切片得到低维张量与低维张量反向填充得到高维张量
        """
        detector_act = self.get_activation(v)
        batch_size, ch, ar, ac = detector_act.shape
        
        #求mx
        mx = None
        cur_parts = []
        for i in xrange(self.pool_shape[0]):
            cur_parts.append([])
            for j in xrange(self.pool_shape[1]):
                cur = detector_act[:, :, i:ar:self.pool_shape[0], j:ac:self.pool_shape[1]]
                if mx is None:
                    mx = cur
                else:
                    mx = T.maximum(cur, mx)
                cur_parts[-1].append(cur)
                
        #求safe_exp和分母切片deno
        pool_term = T.exp(-mx)
        deno = pool_term
        safe_parts = []
        for i in xrange(self.pool_shape[0]):
            safe_parts.append([])
            for j in xrange(self.pool_shape[1]):
                cur_safe = T.exp(cur_parts[i][j] - mx)
                deno = deno + cur_safe
                safe_parts[-1].append(cur_safe)
                
        off_prop = pool_term / deno # 计算出pool层的抽样概率矩阵
        p_pool = 1. - off_prop
                
        #求所有的概率切片矩阵，仍然要考虑空间构型
        h_parts = []
        for i in xrange(self.pool_shape[0]):
            h_parts.append([]) # 仍然维持几何形状
            for j in xrange(self.pool_shape[1]):
                cur_h = safe_parts[i][j] / deno
                h_parts[-1].append(cur_h)
        
        #将切片返回填充4维张量, 得到h的概率密度矩阵(其实是个张量)
        p_h = T.alloc(0., batch_size, ch, ar, ac) # 初始化一个空的4D-tensor作为预填充模板
        for i in xrange(self.pool_shape[0]):
            for j in xrange(self.pool_shape[1]):
                p_h = T.set_subtensor(p_h[:, :, i:ar:self.pool_shape[0], j:ac:self.pool_shape[1]], h_parts[i][j])
        h_mean = p_h
        pool_mean = p_pool
        if sample is False: #如果只需要概率密度矩阵，则返回        
            return [h_mean, pool_mean]
        
        #进行抽样
        events = [] # 建立一维列表存储隐层概率密度切片
        for i in xrange(self.pool_shape[0]):
            for j in xrange(self.pool_shape[1]):
                events.append(h_parts[i][j])
        events.append(off_prop)
        events = [event.dimshuffle(0,1,2,3,'x') for event in events] #新增轴向
        events = tuple(events)
        
        stacked_events = T.concatenate(events, axis=4) #沿新增轴向拼接成一个新的5D-tensor
        rows = ar // self.pool_shape[0] #每个feature_map的行方向上有多少个pool区块
        cols = ac // self.pool_shape[1] #每个feature_map的列方向上有多少个pool区块
        
        outcomes = self.pool_shape[0] * self.pool_shape[1] + 1
        reshaped_events = stacked_events.reshape((batch_size * rows * cols * ch, outcomes))
        
        multinomial = self.theano_rng.multinomial(pvals=reshaped_events, dtype=theano.config.floatX)
        reshaped_multinomial = multinomial.reshape((batch_size, ch, rows, cols, outcomes))
        
        h_sample = T.alloc(0., batch_size, ch, ar, ac) #预制空白模板
        idx = 0 # 起始时对5D-tensor reshaped_multinomial的最后一个轴向切分时的起始索引
        for i in xrange(self.pool_shape[0]):
            for j in xrange(self.pool_shape[1]):
                #填充模板
                h_sample = T.set_subtensor(h_sample[:, :, i:ar:self.pool_shape[0], j:ac:self.pool_shape[1]], reshaped_multinomial[:, :, :, :, idx])
                idx = idx + 1 #切片位置移向下一个位置
        
        pool_sample = 1 - reshaped_multinomial[:, :, :, :, -1]
        
        return [h_mean, h_sample, pool_mean, pool_sample]

    #计算转置卷积，即计算visible_layer收到的来自隐层的输入信号
    def get_activation_T(self, h):
        """
        v: 4D-tensor
        本方法的意义在于，p(h|v)是一个sigmoid函数，对于隐单元h_k_mn，其激活函数为activation的第k个feature map的索引为(m, n)单元的值
        """
        conv_activation_v = self.transformer.lmul_T(h) # activation也是4D-tensor
        activation_v = conv_activation_v + self.v_bias.dimshuffle(0, 'x', 'x')
        return activation_v
    
    #模块4：计算p(v|h)的概率密度矩阵，输入参数h是一个4D-tensor
    def propdown(self, h):
        """利用性质：
        #T.sum(lmul(v) * h, axis=[1,2,3]) = T.sum(lmul_T(h) * v, axis=[1,2,3]), 左侧是标准的能量函数记法
        #因此右侧是能量函数的另外一种表达式，而根据右侧表达式，可以很容易的计算出p(v|h)的sigmoid函数的激活函数"""
        activation_v = self.get_activation_T(h) #activation_v是一个和输入v形状相同的4D-tensor
        #batch_size, ch, ar, ac = activation_v.shape
        v_mean = T.nnet.sigmoid(activation_v) #由于sigmoid函数是element-wise的，所以不需要reshape，可以直接以4D张量的形式作为抽样矩阵

        return v_mean

    #模块5：利用p(v|h)抽样v，输入h是一个4D-tensor
    def sample_v_given_h(self, h):
        ''' This function infers state of visible units given hidden units '''
        v_mean = self.propdown(h)
        v_sample = self.theano_rng.binomial(size=v_mean.shape,
                                             n=1, p=v_mean,
                                             dtype=theano.config.floatX)
        return [v_mean, v_sample]

    def gibbs_hvh(self, h0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the hidden state'''
        v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        h1_mean, pool1_mean, h1_sample, pool1_sample = self.sample_hp_given_v(v1_sample, sample=True)
        return [v1_mean, v1_sample, h1_mean, h1_sample, pool1_mean, pool1_sample]

    def gibbs_vhv(self, v0_sample):
        ''' This function implements one step of Gibbs sampling,
            starting from the visible state'''
        h0_mean, h0_sample, pool0_mean, pool0_sample = self.sample_hp_given_v(v0_sample, sample=True)
        v1_mean, v1_sample = self.sample_v_given_h(T.cast(h0_sample, dtype=theano.config.floatX))
        return [h0_mean, h0_sample, pool0_mean, pool0_sample, v1_mean, v1_sample]
    
    def get_default_cost(self):
        return MyCD_energy_scan(k=1)
				
	# interface for pylearn2.model.mlp PretraindLayer
    def upward_pass(self, state_below):
        return self.propup(state_below)[1]
    
if __name__ == '__main__':
    
    from pylearn2.datasets.mnist import MNIST
    from pylearn2.training_algorithms.sgd import SGD
    from pylearn2.train import Train
    from pylearn2.termination_criteria import MonitorBased
    from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
    from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
    from pylearn2.training_algorithms.sgd import MomentumAdjustor
    from pylearn2.termination_criteria import EpochCounter
	
	
    dsm_train = MNIST(which_set='train', start=0, stop=50000, one_hot=True)
    dsm_valid = MNIST(which_set='train', start=50000, stop=60000, one_hot=True)
    dsm_test = MNIST(which_set='test', start=0, stop=10000, one_hot=True)
	
    monitoring_dataset = {'train': dsm_train, 'valid': dsm_valid, 'test': dsm_test}
    #monitoring_dataset = {'train': dsm_train}
	
    isp = Conv2DSpace(shape=(28, 28), num_channels=1, axes=('b', 'c', 0, 1))
    rbm_model = MyConvRBM(input_space=isp, output_channels=1, pool_shape=(4, 4), batch_size=None, detector_axes=('b', 'c', 0, 1), 
                    kernel_shape=(5, 5), kernel_stride=(1, 1), border_mode='valid', 
                    transformer=None, h_bias=None, v_bias=None, numpy_rng=None,theano_rng=None)
    
    alg = SGD(learning_rate=0.01, cost=None, batch_size=20, init_momentum=None, monitoring_dataset=monitoring_dataset,
              termination_criterion=EpochCounter(max_epochs=20))
    
    train = Train(dataset=dsm_train, model=rbm_model, algorithm=alg, save_path='my_crbm_trainsave.pkl', save_freq=10)
    
    train.main_loop()
    

