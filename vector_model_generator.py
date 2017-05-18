#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wang Haoyu'

import os
import re
import codecs
import full2half
import jieba
import time
import logging
from gensim.models import word2vec

def read_file(filename):
    with open(filename, 'r', errors='ignore') as f:
        lines = [
            line.strip() for line in f.readlines()
        ]
    return lines

def read_sw():
    data_path = os.path.join(os.path.abspath('.'), 'Data')
    sw_path = os.path.join(data_path, 'stopwords_cn.txt')
    ret_list = [line.strip('\n') for line in read_file(sw_path)]
    for x in ['a', 'A']:
        ret_list += [chr(ord(x) + i) for i in range(0, 26)]
    return ret_list + ['/', u'\u3000']

def parse_text(text, sw_list):
    seg_list = jieba.cut(text)
    new_list = []
    for w in seg_list:
        w = full2half.f2h_ch(w)
        if w in sw_list:
            continue
        new_list.append(w)
    return ' '.join(new_list)

def generate_new_file():
    data_path = os.path.join(os.path.abspath('.'), 'Data')
    pattern = re.compile(r'<content>(.*?)</content>')
    sw_list = read_sw()

    corpus_path = os.path.join(data_path, 'corpus')
    if os.path.exists(corpus_path) == False:
        os.mkdir(corpus_path)

    cur_path = os.path.join(data_path, 'SogouCA')
    file_list = os.listdir(cur_path)
    file_list.sort()
    text_count = 0
    for fn in file_list:
        abspath = os.path.join(cur_path, fn)
        #text = read_file(abspath)
        print('Processing text %s...' % fn)
        with open(os.path.join(corpus_path, 'corpus_%s.dat' % text_count), 'w', encoding='UTF-8') as cf:
            with open(abspath, 'r', errors='ignore') as text_file:
                for line in text_file:
                     mt = pattern.match(line)
                     if mt:
                         seg_text = parse_text(mt.group(1), sw_list)
                         cf.write(seg_text)
        text_count += 1

def merge_file():
    data_path = os.path.join(os.path.abspath('.'), 'Data')
    corpus_path = os.path.join(data_path, 'corpus')
    merge_path = os.path.join(data_path, 'merge_corpus')
    if os.path.exists(merge_path) == False:
        os.mkdir(merge_path)
    fn_dir = os.listdir(corpus_path)
    merge_name = 'corpus_merge.dat'
    with codecs.open(os.path.join(merge_path, merge_name), 'w', 'UTF-8') as mf:
        for fn in fn_dir:
            with open(os.path.join(corpus_path, fn), 'rb') as cf:
                seg = cf.read().decode('UTF-8')
                mf.write(seg)
                print('%s done.' % fn)

def modeling():
    data_path = os.path.join(os.path.abspath('.'), 'Data')
    merge_path = os.path.join(data_path, 'merge_corpus')
    merge_name = 'corpus_merge.dat'
    logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
    text = word2vec.Text8Corpus(os.path.join(merge_path, merge_name))
    model = word2vec.Word2Vec(text, size = 400)

    model.save(os.path.join(data_path, 'corpus.model'))
    model.wv.save_word2vec_format(os.path.join(data_path, 'corpus.model.bin'), binary=True)



