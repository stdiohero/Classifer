#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wang Haoyu'

import train_and_test as tnt
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import manifold

# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    color = ('black', 'r')
    shape = ['o', '+']

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], shape[int(y[i])],
                 color=color[int(y[i])],
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

def use_t_SNE(train_file, test_file, svc_model_name):
    ## Loading and curating the data
#    clf, X_reduced, y_reduced = tnt.get_test_clfXy(train_file, test_file, svc_model_name)
    X_reduced, y_reduced, X = tnt.get_train_Xy(train_file)
#    X = tnt.get_origin_X(train_file, test_file, svc_model_name)
#    y = clf.predict(X_reduced)

    # t-SNE embedding of the digits dataset
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)

    plot_embedding(X_tsne, y_reduced, "t-SNE embedding (time %.2fs)" % (time() - t0))
    plt.show()

def plot_pie_chart(labels, sizes, colors, explode):
    #调节图形大小，宽，高
    plt.figure(figsize=(6,6))

    patches,l_text,p_text = plt.pie(sizes,explode=explode,labels=labels,colors=colors,
                                    labeldistance = 1.1,autopct = '%3.1f%%',shadow = False,
                                    startangle = 90,pctdistance = 0.6)
    #改变文本的大小
    #方法是把每一个text遍历。调用set_size方法设置它的属性
    for t in l_text:
        t.set_size=(30)
    for t in p_text:
        t.set_size=(20)
    # 设置x，y轴刻度一致，这样饼图才能是圆的
    plt.axis('equal')
    plt.legend()
    plt.show()

def use_pie_chart(train_file, test_file, svc_model_name):
    y_predict = tnt.get_predict_result(train_file, test_file, svc_model_name)
    pos_cnt = 0
    neg_cnt = 0
    for x in y_predict:
        if x == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    labels = [u'positive', u'negative']
    sizes = [pos_cnt, neg_cnt]
    colors = ['red','blue']
    explode = (0.05,0)
    plot_pie_chart(labels=labels, sizes=sizes, colors=colors, explode=explode)


#use_pie_chart('ChnSentiCorp_htl_unba_10000','ChnSentiCorp_test', 'ChnSentiCorp_svc.pkl')
#use_pie_chart('xiyou_movie', 'test_xiyou_movie', 'xiyou_movie_svc.pkl')
#use_t_SNE('ChnSentiCorp_htl_unba_10000','ChnSentiCorp_test', 'ChnSentiCorp_svc.pkl')
#use_t_SNE('xiyou_movie', 'test_xiyou_movie', 'xiyou_movie_svc.pkl')
