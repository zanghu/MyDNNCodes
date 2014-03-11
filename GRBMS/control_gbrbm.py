#coding=utf8
#模型训练控制文件示例
if __name__ == '__main__':
    
    from new_GBRBM import MyGaussianBinaryRBM, MyCD_scan, HonglakLeeSparse
    from pylearn2.datasets.mnist import MNIST
    from pylearn2.training_algorithms.sgd import SGD
    from pylearn2.train import Train
    from pylearn2.termination_criteria import MonitorBased
    from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
    from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
    from pylearn2.training_algorithms.sgd import MomentumAdjustor
    from pylearn2.termination_criteria import EpochCounter
    from pylearn2.costs.cost import SumOfCosts
    import time
    import cPickle

	
    f = open('/home/zanghu/useful_data/dsc_cifar10_preprocessed_patches.pkl')
    dsc_train = cPickle.load(f)
    f.close()
    dsc_train.X = dsc_train.X[0:1000, :]
    #monitoring_dataset = {'train': dsm_train, 'valid': dsm_valid, 'test': dsm_test}
    monitoring_dataset = {'train': dsc_train}
	
    rbm_model = MyGaussianBinaryRBM(n_vis=192, n_hid=400)
    
    #cd_cost = MyCD_scan(k=15)
    #total_cost = MyPCD_scan(k=15, chain_num=20)
    total_cost = SumOfCosts(costs=[MyCD_scan(k=1), HonglakLeeSparse(p=0.02)])
    
    alg = SGD(learning_rate=0.001, cost=total_cost, batch_size=20, init_momentum=None, monitoring_dataset=monitoring_dataset,
              #termination_criterion=MonitorBased(channel_name='valid_recon_error', N=10))
              termination_criterion=EpochCounter(max_epochs=15))  
    
    #MonitorBasedLRAdjuster(dataset_name='valid'),MomentumAdjustor(start=1, saturate=20, final_momentum=.99)
    train = Train(dataset=dsc_train, model=rbm_model, algorithm=alg,
            #extensions=[MonitorBasedSaveBest(channel_name='test_recon_error', save_path='my_rbm_1021.pkl')],
            save_path='my_rbm_trainsave_1021.pkl',
            save_freq=10)
    t0 = time.clock()
    train.main_loop()
    t1 = time.clock()
    print 'time elapsed on training is', t1 - t0

