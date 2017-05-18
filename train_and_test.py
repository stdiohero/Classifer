#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wang Haoyu'

import vectoring
import os
import math
import random
#import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
#from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc


def train_svm(train_file, svc_model_name, model = None, C=10, is_balanced=1, gamma='auto'):
    if model == None:
        model = vectoring.load_model()

    clf = vectoring.train_svm(model, train_file, C=C, is_balanced=is_balanced, gamma=gamma)
    joblib.dump(clf, svc_model_name)
    return clf

def get_train_Xy(train_file):
    return vectoring.get_train_Xy(train_file)

def get_test_clfXy(train_file, test_file, svc_model_name, C=10, is_balanced=1, gamma='auto'):
    model = vectoring.load_model()
    cur_path = os.path.abspath('.')
    if os.path.exists(os.path.join(cur_path, svc_model_name)):
        clf = joblib.load(svc_model_name)
    else:
        clf = train_svm(train_file, svc_model_name, model, C=C, is_balanced=is_balanced, gamma=gamma)

    test_neg_input = vectoring.build_vec(test_file, 'neg', model)
    test_pos_input = vectoring.build_vec(test_file, 'pos', model)

    X_test, y_test = vectoring.generate_xy(test_neg_input, test_pos_input)

    X_reduced_test = PCA(n_components=100).fit_transform(X_test)
#    y_reduced_test = PCA(n_components=100).fit_transform(y_test)
    y_reduced_test = y_test
    y_reduced_test = y_reduced_test.reshape(len(y_reduced_test), 1)

    return clf, X_reduced_test, y_reduced_test

def get_test_Xy_without_clf(test_file):
    model = vectoring.load_model()

    test_neg_input = vectoring.build_vec(test_file, 'neg', model)
    test_pos_input = vectoring.build_vec(test_file, 'pos', model)

    X_test, y_test = vectoring.generate_xy(test_neg_input, test_pos_input)

    X_reduced_test = PCA(n_components=100).fit_transform(X_test)
    y_reduced_test = y_test
    y_reduced_test = y_reduced_test.reshape(len(y_reduced_test), 1)

    return X_reduced_test, y_reduced_test


def get_origin_X(train_file, test_file, svc_model_name):
    model = vectoring.load_model()
    cur_path = os.path.abspath('.')
    if os.path.exists(os.path.join(cur_path, svc_model_name)):
        clf = joblib.load(svc_model_name)
    else:
        clf = train_svm(train_file, svc_model_name, model)

    test_neg_input = vectoring.build_vec(test_file, 'neg', model)
    test_pos_input = vectoring.build_vec(test_file, 'pos', model)

    X_test, y_test = vectoring.generate_xy(test_neg_input, test_pos_input)
    return X_test

def get_predict_result(train_file, test_file, svc_model_name):
    clf, X_reduced_test, y_reduced_test = get_test_clfXy(train_file, test_file, svc_model_name)
    y_predict = clf.predict(X_reduced_test)
    return y_predict

def test_svm(train_file, test_file, svc_model_name, C=10, is_balanced=1, gamma='auto'):
    clf, X_reduced_test, y_reduced_test = get_test_clfXy(train_file, test_file, svc_model_name,
                                                         C=C, is_balanced=is_balanced, gamma=gamma)

#    print('Test Accuracy: %.2f'% clf.score(X_reduced_test, y_reduced_test))
    pred_probas = clf.predict_proba(X_reduced_test)[:,1]

    # plot ROC curve
    fpr,tpr,_ = roc_curve(y_reduced_test, pred_probas, pos_label=1)
    roc_auc = auc(fpr,tpr)
    plt.rcParams['font.sans-serif']=['SimHei']     #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号
    plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc = 'lower right')
    plt.title('ROC分析')
    plt.show()

def grid_search_by_auc(train_file, test_file, svc_model_name):
    C_list = [2, 10, 20]
    gamma_list = [0.001, 0.0001]
    is_balanced = 1
    best_cng = [C_list[0], gamma_list[0]]
    best_auc = 0
    best_clf = None
    X_reduced_test, y_reduced_test = get_test_Xy_without_clf(test_file)
    model = vectoring.load_model()
    cnt = 0

    for C in C_list:
        for gamma in gamma_list:
            cnt += 1
            clf = vectoring.train_svm(model, train_file, C=C, is_balanced=is_balanced, gamma=gamma)
            pred_probas = clf.predict_proba(X_reduced_test)[:,1]
            fpr,tpr,_ = roc_curve(y_reduced_test, pred_probas, pos_label=1)
            roc_auc = auc(fpr,tpr)
            print("%d# C=%d, gamma=%s, auc=%.2f" % (cnt, C, gamma, roc_auc))
            if roc_auc >= best_auc:
                best_auc = roc_auc
                best_clf = clf
                best_cng[0] = C
                best_cng[1] = gamma

    joblib.dump(best_clf, svc_model_name)
    return best_auc, best_cng


def plot_scatter(train_file, test_file, svc_model_name):
    def plot_point_in_circle(radius, X0, Y0, color, marker):
        theta = random.random()*2*np.pi
        r = random.uniform(0, radius)
        x = math.sin(theta)* (r**0.5)
        y = math.cos(theta)* (r**0.5)
        return plt.scatter(x+X0, y+Y0, marker=marker, c = color, s=20, alpha=0.45)

    clf, X_reduced_test, y_reduced_test = get_test_clfXy(train_file, test_file, svc_model_name)

    predict_y = clf.predict(X_reduced_test)
    colors = ['r', 'b']
    marker = ['o', '^']
    center = [(4, 4), (16, 16)]
    radius = 64

    plt.figure(figsize=(8, 8))
    plt.title('分类结果散点图表示')
    plt.rcParams['font.sans-serif']=['SimHei']     #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号
    tp = []
    tp.append(None)
    tp.append(None)
    for index, y in enumerate(predict_y):
        X0 = center[int(y)][0]
        Y0 = center[int(y)][1]
        y_true = int(y_reduced_test[index])
        tp[y_true] = plot_point_in_circle(radius=radius, X0=X0, Y0=Y0, color=colors[y_true], marker=marker[y_true])

    plt.plot([-2, 22], [22, -2], 'k--')
    plt.legend(handles=[tp[0], tp[1]], labels=['negative', 'positive'], loc='best')
    plt.show()



    """
def test_svm_sbs():
    #单独识别还不会写，因为PCA对(n,m)矩阵降维到(n,k)需要满足k<m以及k<=n,所以暂时无法对小于100的数据量进行处理
    #这个方法可以运行，但是无法提取输入的数据产生的结果，所以暂时作废

    model = vectoring.load_model()
    default_input = vectoring.build_vec('ChnSentiCorp_default', '', model)
    clf = joblib.load('SVC.pkl')
    print("请输入文本:")
    str = input()
    print('正在分析中...')
    str = str.strip()
    seg_list = jieba.cut(str)
    vec_list = vectoring.get_vec(list(seg_list), model)

    X_test = default_input
    X_test.append(sum(np.array(vec_list))/len(vec_list))
    X_test = np.array(X_test)
    X_test = scale(X_test)

    X_reduced_test = PCA(n_components=100).fit_transform(X_test)
    print(X_reduced_test.shape)

    print('分析完毕。结果如下：')
    print(clf.predict(X_reduced_test))

    """

def plot_ROC(train_file, test_file, svc_model_name, C=10, is_balanced=1, gamma='auto'):
    test_svm(train_file, test_file, svc_model_name, C=C, is_balanced=is_balanced, gamma=gamma)
#        test_svm('xiyou_movie', 'test_xiyou_movie', 'xiyou_movie_svc.pkl')
#        test_svm('ChnSentiCorp_htl_unba_10000','ChnSentiCorp_test', 'ChnSentiCorp_svc.pkl')

#plot()
#plot_scatter('ChnSentiCorp_htl_unba_10000','ChnSentiCorp_test', 'ChnSentiCorp_svc.pkl')
#plot_scatter('xiyou_movie', 'test_xiyou_movie', 'xiyou_movie_svc.pkl')
#print(grid_search_by_auc('ChnSentiCorp_htl_unba_10000','ChnSentiCorp_test2', 'ChnSentiCorp_svc_grid_search.pkl'))
#print(grid_search_by_auc('xiyou_movie', 'test_xiyou_movie2', 'xiyou_movie_svc_grid_search.pkl'))
#print(grid_search_by_auc('xiyou_movie_all', 'test_xiyou_movie2', 'all_xiyou_movie_svc_grid_search.pkl'))

