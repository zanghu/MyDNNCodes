#coding: utf8
import os
from itertools import product

orin = []
#f = open('/home/zanghu/实验save/multimodal/AEs/contrast_multiAE
f = open('/home/zanghu/实验save/multimodal/AEs/contrast_multiAE/control_ae.py')
for line in f:
    orin.append(line)
f.close()

prefix = '/home/zanghu/实验save/multimodal/AEs/contrast_multiAE/exp_scripts/control_ae_'
lr_list = ['0.1', '0.09', '0.08', '0.07', '0.06', '0.05', '0.04', '0.03', '0.02', '0.01'] #共10个lr
epoch_list = ['5', '10', '15', '30', '50', '75', '100', '125', '150', '225'] #共10组epoch
h_list = ['300', '400', '500', '750', '1000'] #共5组隐单元数
for lr, epoch, h_num in product(iter(lr_list), iter(epoch_list), iter(h_list)):
    f = open(prefix + 'lr' + lr + 'epoch' + epoch + 'h' + h_num + '.py', 'w')
    for i, t in enumerate(orin):
        wrote = False
        if i == 21:
            f.write("    ae_model = MyAutoEncoder(n_vis=784, n_hid=" + h_num + ", corruptor=None)\n")
            wrote = True
        if i == 23:
            f.write("    alg = SGD(learning_rate=" + lr + ", cost=None, batch_size=20, init_momentum=None, monitoring_dataset=monitoring_dataset, termination_criterion=EpochCounter(max_epochs=" + epoch + "))\n")
            wrote = True
        if i == 25:
            f.write("    train = Train(dataset=dsm_train, model=ae_model, algorithm=alg, save_path='ae_save_" + 'lr' + lr + 'epoch' + epoch + 'h' + h_num + ".pkl', save_freq=5)\n")
            wrote = True
        if wrote is False:
            f.write(t)
    f.close()
