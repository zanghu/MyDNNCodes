#coding=utf-8
import os
import time
import numpy

from module.show_text import make_doc

def make_stem_text(design_matrix_path, freq_stem_path, prefix='train'):
    """
    需要读入的文件: 词向量设计矩阵，freq_stem(2000高频词干) #final_dict(原始词干字典的序号：2000高频词干字典的序号),
    把每个词向量，先转化为类似20newsgroups数据集的train.data和test.data的形式，之后将freq_stem做成vocabulary.txt的形式
    最后使用show_test.py来制作“词干：词频”文档
    """
    f = open(design_matrix_path, 'r')
    design_matrix = numpy.load(design_matrix_path)
    f.close()
    
    f = open(prefix + '_bow.data', 'w')
    for i in xrange(design_matrix.shape[0]):
        for j in xrange(design_matrix.shape[1]):
            if design_matrix[i][j] != 0:
                f.write(' '.join([str(i+1), str(j+1), str(int(design_matrix[i][j]))])) #注意序号等于索引号+1
                f.write('\n')
    f.close() 
    
    f1 = open(freq_stem_path, 'r')
    f2 = open('stem_voc.txt', 'w')
    cnt = 0
    for line in f1:
        w = line.split(':')[0]
        f2.write(w + '\n')
        cnt += 1
    assert cnt == 2000
    f1.close()
    f2.close()

    make_doc(doc_type=prefix, doc_number=18874, stop_num=None, each_num=200, data_path=prefix + '_bow.data', volcabulary_path='stem_voc.txt')

if __name__ == '__main__':
    make_stem_text(design_matrix_path='py_20news_train.npy', freq_stem_path='freq_stem.txt', prefix='train')
    make_stem_text(design_matrix_path='py_20news_test.npy', freq_stem_path='freq_stem.txt', prefix='test')

 
