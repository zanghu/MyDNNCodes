#coding=utf-8                                                                                                                                                                                                                                            
"""
本文件的作用：将20newsgroups的train.data或test.data转换为"单词: 词频"的形式，以便查看
需要两个文件：
1.词典单词表，例如volcabulary.txt
2.需要生成文档的数据文件，例如train.data
"""
import os
import time
import numpy

def make_prefix(n):
    """"""
    s = ''
    l = []
    temp = 1.
    while n / temp >= 1:
        temp *= 10.
        l.append(s)
        s = s + '0' 
    return l
                                                                                                             
def make_doc(doc_type, doc_number, stop_num=None, each_num=200, data_path='train.data', volcabulary_path='vocabulary.txt'):
    """
    Parameters
    ----------
    doc_type: str，可选'train'或'test'，用于给保存目录命名前缀
    doc_number: int, train.data的总文档数
    stop_num: int，optional, 希望生成的前若干个文档数
    each_num: int, optional, 每个子文件夹中的文档数，建议根据总文档数大小选择100-500，如果值过大，子文件夹打开速度会变慢，反之则子目录太多
    """
    assert isinstance(doc_type, str)
    save_path = doc_type + '_text'
    if stop_num is None:
        stop_num = doc_number

    prefix = make_prefix(doc_number) 
    print "prefix is:", prefix

    #读入词典
    word_dict1 = {} #词：序号
    #word_dict2 = {} #序号：词
    f = open(volcabulary_path, 'r')
    for n, line in enumerate(f):
        w = line.split('\n')[0]
        word_dict1[w] = n + 1 #词：该词在当前字典中的序号，注意python中编号从0开始
    f.close()
    word_dict2 = {v:k for k, v in word_dict1.items()} #字典键值互换

    if os.path.exists(save_path):
        pass
        #if os.listdir('./text/') != []:
            #print os.listdir('./text/')
            #print '默认文件保存路径已经存在，请指定其他路径'
    else:
        os.mkdir(save_path)

    cur_text = {}
    cur_idx = 1
    f = open(data_path, 'r')

    for line in f:

        if cur_idx > stop_num:
            break

        l_list = line.strip('\n').split(' ')
        doc_idx = eval(l_list[0]) #文档编号
        w_num = eval(l_list[1]) #词序号
        cnt = eval(l_list[2]) #词频，其实不转化维数值也无所谓

        #建立子文件夹
        if doc_idx % each_num == 1:
            cur_save_path = os.path.join(save_path, str(doc_idx) + '-' + str(doc_idx + each_num))
            if not os.path.exists(cur_save_path):
                os.mkdir(cur_save_path)

        if doc_idx != cur_idx:
            #print "current num:", len(prefix) - len(l_list[0])
            pre = prefix[len(prefix) - len(l_list[0])] #补零
            t = open(os.path.join(cur_save_path, pre + str(cur_idx) + '.txt'), 'w')
            for key, value in cur_text.items():
                w = word_dict2[key]
                t.write(w + ':' + str(value) + '\n')
            t.write('\n')
            t.close()
            cur_idx += 1 #更新状态标记维下一个文档
            cur_text.clear() #清空字典

        cur_text[w_num] = cnt #插入当前文本记录
    f.close()

if __name__ == '__main__':
    #make_doc(doc_type='train', doc_number=18874, stop_num=None)
    make_doc(doc_type='test', doc_number=7505, stop_num=None)
