#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wang Haoyu'

import os
#import sys
#import codecs
import jieba
#import logging
#import re
import numpy as np
import matplotlib.pyplot as plt
import gensim
#from gensim.models import word2vec
#from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.svm import SVC
#from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
#from sklearn.metrics import roc_curve, auc
#from scipy import stats



def parse_seg(seg):
    word_list = jieba.cut(seg)
    return list(word_list)

def read_file(dir_name, neg_or_pos):
    data_path = os.path.join(os.path.abspath('.'), 'Data')
    dir_path = os.path.join(data_path, dir_name)
    if neg_or_pos != '':
        dir_path = os.path.join(dir_path, neg_or_pos)
    file_list = os.listdir(dir_path)
    outputs = []
    for fn in file_list:
        with open(os.path.join(dir_path, fn), 'r', errors='ignore') as f:
            seg = parse_seg(f.read().strip())
            outputs.append(seg)
    return outputs

def get_vec(word_list, model):
    word_vec = []
    for word in word_list:
        word = word.replace('\n', '')
        try:
            word_vec.append(model[word])
        except KeyError:
            continue
    #self.__word_vec = np.concatenate(self.__word_vec)
    return np.array(word_vec, dtype=float)

def build_vec(dir_name, neg_or_pos, model):
    output = []
    text_file = read_file(dir_name, neg_or_pos)
    for word_list in text_file:
        vec_list = get_vec(word_list, model)
        if len(vec_list) != 0:
            output.append(sum(np.array(vec_list))/len(vec_list))
    return output


def generate_xy(neg_input, pos_input):
    #1代表正向情感，0代表负向情感
    y = np.concatenate((np.ones(len(pos_input)), np.zeros(len(neg_input))))
    X = pos_input[:]
    for neg in neg_input:
        X.append(neg)
    X = np.array(X)
    return scale(X), y

def load_model():
    data_path = os.path.join(os.path.abspath('.'), 'Data')
    model_path = os.path.join(data_path, 'corpus.model.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model

def get_train_Xy(filename, model=None):
    if model == None:
        model = load_model()
    pos_input = build_vec(filename, 'pos', model)
    neg_input = build_vec(filename, 'neg', model)

    X, y = generate_xy(neg_input, pos_input)
    X_reduced_trained = PCA(n_components=100).fit_transform(X)
    #    Y_reduced_trained = PCA(n_components=100).fit_transform(y)
    Y_reduced_trained = y
    Y_reduced_trained = Y_reduced_trained.reshape(len(Y_reduced_trained), 1)
    return X_reduced_trained, Y_reduced_trained, X

def train_svm(model, filename, C=10, is_balanced=1, gamma='auto'):
    pos_input = build_vec(filename, 'pos', model)
    neg_input = build_vec(filename, 'neg', model)

    X, y = generate_xy(neg_input, pos_input)
    X_reduced_trained = PCA(n_components=100).fit_transform(X)
    #    Y_reduced_trained = PCA(n_components=100).fit_transform(y)
    Y_reduced_trained = y
    Y_reduced_trained = Y_reduced_trained.reshape(len(Y_reduced_trained), 1)

    if is_balanced == 1:
        clf = SVC(C = C, probability=True, gamma=gamma, cache_size=1000, class_weight='balanced')
    else:
        clf = SVC(C = C, probability=True, gamma=gamma, cache_size=3000)

    clf.fit(X_reduced_trained, Y_reduced_trained)
    return clf

def K_fold_cv(filename, model=None):
    if model == None:
        model = load_model()
    pos_input = build_vec(filename, 'pos', model)
    neg_input = build_vec(filename, 'neg', model)

    X, y = generate_xy(neg_input, pos_input)
    X_reduced_trained = PCA(n_components=100).fit_transform(X)
    Y_reduced_trained = y
#    Y_reduced_trained = Y_reduced_trained.reshape(len(Y_reduced_trained), 1)

    clf = SVC(C = 10, probability=True, gamma='auto', cache_size=1000, class_weight='balanced')
    scores = cross_validation.cross_val_score(clf, X_reduced_trained, Y_reduced_trained, cv=10)
    return scores

def grid_search_CV(filename, model=None):
    if model == None:
        model = load_model()
    pos_input = build_vec(filename, 'pos', model)
    neg_input = build_vec(filename, 'neg', model)

    X, y = generate_xy(neg_input, pos_input)
    X_reduced_trained = PCA(n_components=100).fit_transform(X)
    Y_reduced_trained = y
#    Y_reduced_trained = Y_reduced_trained.reshape(len(Y_reduced_trained), 1)

    svc = SVC(cache_size=2500, class_weight='balanced')
    param_grid = {"C": [0.1, 1, 10, 20], "gamma": [0.5, 0.01, 0.001, 'auto']}
    grid = GridSearchCV(svc, param_grid=param_grid, cv=10)
    grid.fit(X_reduced_trained, Y_reduced_trained)

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

def plot_PCA_figure(filename, X=None):
    model = load_model()
    if X == None:
        pos_input = build_vec(filename, 'pos', model)
        neg_input = build_vec(filename, 'neg', model)
        X, y = generate_xy(neg_input, pos_input)

    pca = PCA()
    pca.fit(X)
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])

    plt.plot(pca.explained_variance_, linewidth=2)

    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')

    plt.show()

#test_svm()
#test_svm_sbs()
#plot_PCA_figure( 'xiyou_movie')
#plot_PCA_figure('ChnSentiCorp_htl_unba_10000')
#s = K_fold_cv('ChnSentiCorp_htl_unba_10000')
#s = K_fold_cv( 'xiyou_movie')
#print(s)
#print(sum(s)/len(s))
#grid_search_CV('ChnSentiCorp_htl_unba_10000')
#grid_search_CV('xiyou_movie')
