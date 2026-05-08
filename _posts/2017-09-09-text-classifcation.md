---
layout: post
title: "浅析短文本分类与经验分享"
date: 2017-09-11
excerpt: "summerization of text classifcation"
tags: [naive bayes,svm,fasttext,word2vec]
comments: true
---
本文的应用场景例子是针对**短文本**的。

### 1. 分词  
分词是自然语言处理中最基础的功能，分词功能的好坏对于之后的词法分析和句法分析等尤为重要。同时不同于英文中有空格，中文没有空格，无法直接对句子进行拆分，因此需要分词器对句子进行处理，得到组成句子的最小单元：词，某个文本，实际上就是对某个句子使用部分词进行表达，如"你好,我今天订了手机，什么时候发货呢"，“你好  订了  手机 什么时候  发货”，这里可以看到，像“呢”这种词直接被过滤掉，我们称之为“停止词”，对于文本的表达意义不大。那么对于分词功能的实现来说，其实当下已经有很多的开源工具了，比如[NLPIR汉语分词系统](https://github.com/NLPIR-team/NLPIR)),[结巴分词](https://github.com/huaban/jieba-analysis),基于lucene的IKAnalyzer轻量工具包，等等还有很多，其实效果来说都不会相差太多，看你的需要了，本文考虑到方便和快捷，直接选用的是IKAnalyzer，能自定义词库和停止词词典，能满足我的业务需求。  
### 2. 文本向量化  
#### (1) one-hot +TFIDF  
one-hot是一种比较经典的文本向量表示方法，其思想是把背景语料中的所有词汇作为高维空间中的一维,如单词“手机”，在背景语料中的向量形式为[0,0,0,1,0,0,0,,0,0,0,...]这种形式，出现则置于1，反之置于0，即有一个1以及大量的0组成，对于某句话来说，该句子在背景语料中的句向量便是各个词的向量叠加。那么TFIDF算法（具体细节可看：[http://www.ruanyifeng.com/blog/2013/03/tf-idf.html]()）的作用是什么呢，TFIDF对于比较长的文档来说，能根据词频计算出这片文章比较重要的词汇，但是在短文本的情况下，则可以计算出IDF（逆文档频率），即某个单词在背景语料中的区分度，这样，我们将计算出来的值替换掉原来的1，便更能精确地表示句向量，同时，我们对背景语料的所有出现过的词汇标记一个序号，这样，每个短文本句子可以表示为结构:   **index1:idf1  index2:idf2  index3:idf3 ...**  
以此类推，其中， index为该词在所有词汇中的索引位置，idf为该词的idf值。  
*注：在机器学习算法中可将数值归一化为-1到1之间，便于计算，这里推荐离差标准化如图一所示。*

![](http://owcoclj9i.bkt.clouddn.com/image/jpg/textclassifcation/WechatIMG36.jpg)
<center>图一</center>

![](http://owcoclj9i.bkt.clouddn.com/image/jpg/textclassifcation/pic2.jpg)
<center>图二</center>
![](http://owcoclj9i.bkt.clouddn.com/image/jpg/textclassifcation/pic3.jpg)
<center>图三</center>
####(2)Word2Vec
上述的one-hot表示方法得到的文本向量，有一个很明显的特点，就是稀疏矩阵，一个短文本只含有不多的单词，那么没有出现的则置为0，并不能够充分的表示该文本。于是，出现了另外一种低维、稠密的连续向量，也就是基于深度学习技术实现的分布式表示方法，形式：[0.792, −0.177, −0.107, 0.109, −0.542, …]，维度以200维为比较常见，也就是说，一个短文本如“你好  订了  手机 什么时候  发货”在word2vec则表示为一个长度为200的数组，也即是一个200维的特征向量。值得注意的是，使用深度学习对于数据量比较依赖，需要比较大的数据量才有比较好的效果，本文是在公司其他同事使用python爬取了百度知道近两年的数据，训练了近33个小时得到的2G模型，因此来说效果还不错。一系列短文本使用word2vec直观展示如图所示。  

![](http://owcoclj9i.bkt.clouddn.com/image/jpg/textclassifcation/pic4.jpg)
<center>图四</center>  
上述的结构可以描述为**label1  index1:value1  index2:value2  index3:value3  index4:value4....**一直到200维.其中label为已知的短文本所属类别。 

### 3.分类算法
由于常见的机器学习算法原理等在互联网上资源比较多，本文着重提一些本人在开发过程中遇到的
坑。
#### （1）朴素贝叶斯
核心内容是条件概率，归结到短文本分类中含义是：某个短文本属于某个分类的概率，等于这个类别的概率 乘以 短文本中每个特征词在这个类别的条件概率。  
<center>
![](http://owcoclj9i.bkt.clouddn.com/image/jpg/textclassifcation/pic5.jpg)
</center>  
<center>图五</center>  
在图中公式中，x为所有特征词的集合，y是预测的分类结果也就是标签label，由于p(x)对于所有的label都相等，所以可以只考虑比较上述分子的大小，计算x1 x2 x3..这个特征词组合在所有分类label下的p(y|x)值的大小，最大的那值对应的分类便是预测的结果。  
原理很简单。但有几点需要注意：   

a.由于分子是多个概率值的连乘，如果某个特征词组合一旦有一个单词没有出现在某个分类中，最后求得概率便等于0，于是就找不到分类，解决方法是将默认值设为一个不等于0的数，比如为min =1/（总词数），同时需要设定阈值为min的n次方，n为某个短文本的特征词个数。  

b.因为算法计算会计算每个单词的概率，如果数据量大的话会让程序很缓慢，其他同事也没法进行其他操作了，解决方法是使用java的序列化功能，提前将所有的计算结果以二进制的状态存储为文件，在进行预测的时候，将读模型任务放在构造函数中，这样在预测的时候便只需要读取一次文件即可，经验证，提高了60多倍的程序运行速度。
![](http://owcoclj9i.bkt.clouddn.com/image/jpg/textclassifcation/WechatIMG33.jpg)
![](http://owcoclj9i.bkt.clouddn.com/image/jpg/textclassifcation/WechatIMG34.jpg)
####（2） Libsvm
可以看到，用纯粹朴素贝叶斯算法里没有用到特征向量，而对于封装好的libsvm则不同libsvm是一个svm开源库（[http://www.csie.ntu.edu.tw/~cjlin/libsvm/]()），需将数据处理为上述图三图四格式训练并预测即可,使用libsvm做文本分类可参考博客（[http://shiyanjun.cn/archives/548.html]()）。  
若想拿到工程上应用，也需要注意一些问题。  
a.对于svm，参数调优很重要，对于文本分类，核函数选择RBF，对于C和g，可以利用工具包grid.py进行选择。   
b.libsvm的输入输出都是文件，如果加以应用需要对svm\_train.java和svm\_predict.java进行修改。   
c.输入时短文本，需先计算其特征向量，而使用Word2Vec耗费较大内存，因此在条件限制情况下可直接使用图三特征向量，效果也还可以。	
####（3）Fasttext
Fasttext是facebook针对文本分类的一款开源产品，特点就是快，且性能丝毫不输深度学习。只需要按照其数据格式来即可。**\_\_label\_\_class    word1 word2 word3...  \_\_label\_\_class**为类别，后续word为该短文本包含的单词。使用比较简单，但需要多进行调试，推荐一篇博客[http://blog.csdn.net/luoyexuge/article/details/72677186]()

到此把之前在开发文本分类器中的一些方面简要地写了出来，可能有疏漏和不足之处，欢迎回复讨论。

