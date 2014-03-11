#coding: utf8
if __name__ == '__main__':
    
    from pylearn2.datasets.mnist import MNIST
    from pylearn2.training_algorithms.sgd import SGD
    from pylearn2.costs.cost import SumOfCosts
    from pylearn2.train import Train
    from pylearn2.termination_criteria import MonitorBased
    from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
    from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster
    from pylearn2.training_algorithms.sgd import MomentumAdjustor
    from pylearn2.termination_criteria import EpochCounter
    from module.multi_AE import MyAutoEncoder; import theano; import theano.tensor as T
    
    dsm_train = MNIST(which_set='train', start=0, stop=50000, one_hot=False)
    dsm_valid = MNIST(which_set='train', start=50000, stop=60000, one_hot=False)
    dsm_test = MNIST(which_set='test', start=0, stop=10000, one_hot=False)
	
    monitoring_dataset = {'train': dsm_train, 'valid': dsm_valid, 'test': dsm_test}
    #monitoring_dataset = {'train': dsm_train}
	
    ae_model = MyAutoEncoder(n_vis=784, n_hid=500, corruptor=None)
    #cost=None时，默认代价使用ce-cost(有解码非线性函数)或mse-cost(无解码非线性函数)
    alg = SGD(learning_rate=0.01, cost=None, batch_size=20, init_momentum=None, monitoring_dataset=monitoring_dataset, termination_criterion=EpochCounter(max_epochs=15))
    
    train = Train(dataset=dsm_train, model=ae_model, algorithm=alg, save_path='ae_save.pkl', save_freq=5)
    
    train.main_loop()

    x = T.matrix()
    f = theano.function([x], ae_model.get_enc(x))

    X_propup_train = f(dsm_train.X)

    X_propup_valid = f(dsm_valid.X)

    X_propup_test = f(dsm_test.X)

    from sklearn.linear_model import LogisticRegression

    clf  = LogisticRegression()
    clf.fit(X_propup_train, dsm_train.y)
    print 'on valid-dataset: ', clf.score(X_propup_valid, dsm_valid.y)
    print 'on test-dataset: ', clf.score(X_propup_test, dsm_test.y)
