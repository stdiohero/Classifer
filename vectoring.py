#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wang Haoyu'

import os
import codecs
import jieba
import logging
import re
from collections import defaultdict
from gensim.models import word2vec



class Vectoring(object):
    def __init__(self):
        self.__stop_words = set()
        self.__cur_path = os.path.abspath('.')
        self.__read_sw()

    def __read_sw(self):
        if len(self.__stop_words) != 0:
            return
        sw_path = os.path.join(self.__cur_path, 'Data')
        sw_filename = 'stopwords_cn.txt'
        sw_path = os.path.join(sw_path, sw_filename)
        with open(sw_path, 'r') as f:
            stop_words_list = f.readlines()
        for w in stop_words_list:
            self.__stop_words.add(w.strip('\n'))
        """
        少了换行符和空格，手动添加上
        """
        self.__stop_words.add('\n')
        self.__stop_words.add(' ')

    def get_stopwords(self):
        return self.__stop_words

    def __read_text(self, tx_path):
        with open(tx_path, 'r', errors='ignore') as f:
            re_text = f.read()
        return re_text.strip()

    def text2word(self, text):
        seg_list = jieba.cut(text)
        new_text = []

        for w in seg_list:
            if w in self.__stop_words:
                continue
            new_text.append(w)

        return ' '.join(new_text)   #用空格链接

    def parse_text(self):
        path = os.path.join(self.__cur_path, 'Data')
        path = os.path.join(path, 'ChnSentiCorp_htl_unba_10000')
        path = os.path.join(path, 'neg')
        file_list = os.listdir(path)
        text_list = []
        for fn in file_list:
            text = self.__read_text(os.path.join(path, fn))
            text_list.append(self.text2word(text))
        filename = 'hotel_neg.data'
        n_path = os.path.join(self.__cur_path, 'Data')
        n_path = os.path.join(n_path, filename)
        with codecs.open(n_path, 'w', encoding='UTF-8') as f:
            for t in text_list:
                f.write(t)
        return text_list

    def modeling(self, text_name, model_name):
        logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
        text = word2vec.Text8Corpus(text_name)   #加载语料
        model = word2vec.Word2Vec(text, size=400)   #训练skip-gram模型

        model.save('%s.model' % model_name)  #保存模型
        model.save_word2vec_format('%s.model.bin' % model_name, binary=True)   #格式化保存

    def get_vec(self):
        #TODO
        pass


    def print_sw(self):
        return self.__stop_words


v = Vectoring()
l = v.parse_text()
for s in l:
    print(s)

