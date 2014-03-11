#coding: utf8
import os
from itertools import product

os.chdir('/home/zanghu/multimodal/AEs/contrast_multiAE/exp_scripts/')

prefix = 'control_ae_'
lr_list = ['0.01']#, '0.02', '0.03', '0.04', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1'] #共10个lr
epoch_list = ['5']#, '10', '15', '30', '50', '75', '100', '125', '150', '225'] #共10组epoch
h_list = ['300']#, '400', '500', '750', '1000'] #共5组隐单元数
for lr, epoch, h_num in product(iter(lr_list), iter(epoch_list), iter(h_list)):
    os.system('THEANO_FLAGS=mdoe=FAST_RUN,device=gpu,floatX=float32 python ' + prefix + 'lr' + lr + 'epoch' + epoch + 'h' + h_num + '.py > ../record' + 'lr' + lr + 'epoch' + epoch + 'h' + h_num)
