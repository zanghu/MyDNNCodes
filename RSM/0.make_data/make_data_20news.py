# -*- coding: utf-8 -*-
"""
先去停用词，再词干化. 这是考虑到停用词一般都是常用词，而不是常用词干.另外带有第三人称单数等变形的词一般都是动词，非停用词.
"""
from collections import OrderedDict
import numpy
import cPickle
import time
from nltk import PorterStemmer

def make_design_matrix(matrix_shape, data_path, word_dict, mapping_dict, final_dict):
    """"""
    print '将样本转化为词向量组成的design matrix...'
    num_doc, num_features = matrix_shape
    design_matrix = numpy.zeros(shape=(num_doc, num_features), dtype='float64') #theano.config.floatX
    f = open(data_path)
    #把每个文档组成一个向量
    for line in f:
        wl = line.strip('\n').split(' ')
        if mapping_dict.has_key(eval(wl[1])): # 由于mapping_dict是一句删除停用词后的字典建立的，因此文档中的词不一定都会出现在mapping_dict的key中
            word_num = mapping_dict[eval(wl[1])]
            if final_dict.has_key(word_num): # 检查当前词是否在前2000个高频词之中
                #注意到在python中矩阵的索引从0开始，而字典中单词的编号从1开始
                design_matrix[eval(wl[0])-1, final_dict[word_num]-1] += eval(wl[2])
    f.close()
    #print design_matrix.shape
    print 'design matrix finished...'
    return design_matrix

def make_freq_dict(freq_bound=2000, train_data_path='train.data', voc_path='vocabulary.txt', stop_word_path='engStopWord.dat'):
    """
    制造前2000个高频词字典
    ---------------------
    实质上是制造了前2000个高频词干的词干字典
    步骤：首先，从文档集原始词表中制造词和词序号的映射关系字典
          其次，先从原始字典中去停用词，再对剩下的单词进行词干化，制造词干化字典（词干：词干序号）
          再次，统计样本文档集，得到词干-词干频率字典，对盖子点依value(词频)排序
          最后，取排序最高的前2000个键值对作为高频词(干)字典，返回
    """
    print '您选择使用前%d个高频词(干)作为词向量字典' % freq_bound
    #1.读取单词字典
    f = open(voc_path) #打开训练集词表
    word_dict = {} # 编号-词(字符串)
    check_dict = {} # (字符串)词-编号
    cnt = 1 #词的序号从1开始计数
    for line in f:    
        wl = line.strip('\n').split(' ')
        check_dict[wl[0]] = cnt
        word_dict[cnt] = wl[0]
        cnt += 1
    f.close()
    #print cnt

    #2.删除字典中停用词
    f = open(stop_word_path)
    stop_dict = [] # 停用词列表
    cnt = 0
    for line in f:
        wl = line.strip('\r\n')
        stop_dict.append(wl)
        cnt += 1
    f.close()
    #print cnt
    #print 'len(word_dict)', len(word_dict)

    num_list = []
    non_list=[]
    #找出需要删除的停用词在字典中的编号
    for word in stop_dict :
        if check_dict.has_key(word):
            num_list.append(check_dict[word]) #记录当前停用词在原始词典中的序号
            del word_dict[check_dict[word]]
            del check_dict[word]
        else:
            non_list.append(word)
    print '经使用的停用词个数： %d' % len(num_list)
    print '未使用的停用词个数: %d' % len(non_list)

    #3.词干化，生成新的word_list, check_list和映射表mapping_list，词干的编号从1开始
    word_dict_new = {} #词干的word_dict
    check_dict_new ={} #词干的check_dict
    mapping_dict = {} #词在原始词典(check_dict)的序号: 该词词干在词干字典(check_dict_new)的序号

    stem_cnt = 1
    for word in check_dict.keys():
        stem = PorterStemmer().stem_word(word)
        check_dict_new.setdefault(stem, stem_cnt) #巧妙的使用dict的setdefault()，省去if判断
        stem_cnt += 1
        mapping_dict[check_dict[word]] = check_dict_new[stem] #word的编号映射到词干的编号
    mapping_dict_reverse = {v: k for k, v in mapping_dict.iteritems()} #建立键值互换的映射字典
    print '词干字典建立完成...'

    #4.读取数据，建立词干频率字典
    freq_dict = {} #词干在check_dict_new中的序号: 词干频率
    f = open(train_data_path)
    print '使用训练集统计词频...'
    for line in f:
        wl = line.strip('\n').split(' ')
        if mapping_dict.has_key(eval(wl[1])): #并不是每一个wl[1]都在mapping_dict中，因为有可能该词已经在去停用词阶段被删除了
            stem_num = mapping_dict[eval(wl[1])] 
            #如果word_stem键存在，则下面第一句话不起作用，否则下面两句话一起发生作用
            freq_dict.setdefault(stem_num, 0)
            freq_dict[stem_num] += eval(wl[2])
    f.close()

    print '开始对词频字典排序...'
    t0 = time.clock()
    #对字典按value从大到小进行排序
    ofd = OrderedDict(sorted(freq_dict.iteritems(), key=lambda t: t[1], reverse=True))
    print '排序耗时：%f' % (time.clock() - t0)
    cnt = 1
    final_dict = {} #键是前2000高频词(干)在原始词干字典中的词干序号，值是该词干在高频词干字典字典中的序号(大于1小于等于2000)
    for key, value in ofd.iteritems():
        final_dict[key] = cnt
        cnt += 1
        if cnt > freq_bound:
            break
    print '%d most freq_dict has been established...' % freq_bound

    #记录高频词(干)表以备查阅
    f = open('freq_stem.txt', 'w')
    cnt = 1
    for k, v in ofd.iteritems():
        f.write(word_dict[mapping_dict_reverse[k]] + ': ' + str(v) + '\n')
        cnt += 1
        if cnt > freq_bound:
            break
    f.close()

    return final_dict, mapping_dict

def main():
    """制造词向量组成的设计矩阵"""
    #需要手动设置的参数===========================================================================
    num_doc_train = 11269
    num_doc_test = 7505
    num_features = 2000
    train_data_path = '/home/zanghu/Pro_Datasets/20news-bydate/matlab/train.data'
    train_and_test_data_path = '/home/zanghu/Pro_Datasets/20news-bydate/matlab/train_and_test.data'
    test_data_path = '/home/zanghu/Pro_Datasets/20news-bydate/matlab/test.data'
    #============================================================================================

    #获得两个字典
    final_dict, mapping_dict = make_freq_dict(freq_bound=num_features, train_data_path=train_and_test_data_path)
    #把每个文档组成一个向量
    #获得训练集设计矩阵
    design_matrix_train = make_design_matrix(matrix_shape=(num_doc_train, num_features), data_path=train_data_path, 
                            word_dict=final_dict, mapping_dict=mapping_dict, final_dict=final_dict)
    #获得测试集设计矩阵
    design_matrix_test = make_design_matrix(matrix_shape=(num_doc_test, num_features), data_path=test_data_path, 
                            word_dict=final_dict, mapping_dict=mapping_dict, final_dict=final_dict)
    #储存final_dict
    f = open('final_dict.pkl', 'w')
    cPickle.dump(final_dict, f, protocol=2)
    f.close()
    #储存mapping_dict
    f = open('mapping_dict.pkl', 'w')
    cPickle.dump(mapping_dict, f, protocol=2)
    f.close()

    #保存设计矩阵
    numpy.save('py_20news_train.npy', design_matrix_train)
    #print 'train design matrix saved at: %s' % path_train
    numpy.save('py_20news_test.npy', design_matrix_test)
    #print 'test design matrix saved at: %s' % path_test

if __name__ == '__main__':
    main()
