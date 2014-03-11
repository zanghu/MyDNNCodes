这个文件夹下一共有5个.py文件
运行本文件夹中的文件需要python的nltk模块(也许还有别的模块)

首先，执行0.make_data_20news.py
其次，执行1.make_stem_text.py
    会生成两个文件夹test_stem_text和train_stem_text，里面分别是测试集和训练集的可供阅读的文档
最后，执行2.control_query.py
    会生成一个query的结果分析表quer_error.txt，文件比较大可能会需要很长时间，如果是用文件中的默认参数则会很快

注意：上述第二步中生成的文档中的词都是经过词干化后的词干，如果希望获得去停用词和词干化之前的文档，请单独执行show_text.py，同样会生成两个文件夹，名字是test_text和train_text

本文件夹下的rsm_result_batch001.pkl是一个训练好的rsm模型，你就可以使用你自己训练的rsm模型替换它，这样在执行第三步时，得到的就是你自己模型的结果

本文件夹下的new_query_nade.py是一个针对query类问题的函数文件合集，非常有用，使用了大量矩阵运算因此效率比老版本高很多，具有计算MAP(mean aberage precision，绘制p-r曲线，计算精度和召回率等功能，目前支持三种距离：余弦相似度，调整余弦相似度，欧氏距离）
