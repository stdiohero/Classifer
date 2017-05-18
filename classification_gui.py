#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wang Haoyu'

import train_and_test as tnt
import visualization as visual
import os
import tkinter
from tkinter import *
from tkinter import IntVar
from tkinter import filedialog
from tkinter import ttk



class AddWidget(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_gui()


    def init_gui(self):
        self.root.title('大规模领域文本分类器')
        self.root.option_add('*tearOff', 'FALSE')

        self.grid(column=0, row=0, sticky='nsew')

        self.menubar = tkinter.Menu(self.root)
        self.menu_file = tkinter.Menu(self.menubar)
        self.menu_file.add_command(label='退出', command=self.on_quit)
        self.menubar.add_cascade(menu=self.menu_file, label='文件')

        self.menu_select = tkinter.Menu(self.menubar)
        self.menu_select.add_command(label='酒店评论(10000=3000+7000)', command=self.qs_hotel)
        self.menu_select.add_command(label='电影影评(20000=12000+8000)', command=self.qs_movie)
        self.menu_select.add_command(label='电影影评(70000=35000+35000)', command=self.qs_movie_all)
        self.menubar.add_cascade(menu=self.menu_select, label='快速选择')
        self.root.config(menu=self.menubar)

        self.input_train_file = ttk.Entry(self)
        self.input_train_file.grid(column=1, row=2)
        self.tf_label = ttk.Label(self, text='训练集文件')
        self.tf_label.grid(column=0, row=2, sticky='w')
        self.tf_button = ttk.Button(self, text='选择训练集', command=self.select_train_file)
        self.tf_button.grid(column=2, row=2, sticky='e')

        self.input_model_name = ttk.Entry(self)
        self.input_model_name.grid(column=1, row=3)
        self.mn_label = ttk.Label(self, text='模型名称')
        self.mn_label.grid(column=0, row=3, sticky='w')
        self.mn_button = ttk.Button(self, text='选择模型文件', command=self.select_model_file)
        self.mn_button.grid(column=2, row=3, sticky='e')

        self.input_test_file = ttk.Entry(self)
        self.input_test_file.grid(column=1, row=4)
        self.testF_label = ttk.Label(self, text='测试集文件')
        self.testF_label.grid(column=0, row=4, sticky='w')
        self.testF_button = ttk.Button(self, text='选择测试集', command=self.select_test_file)
        self.testF_button.grid(column=2, row=4, sticky='e')

        self.gamma_label = ttk.Label(self, text='gamma=')
        self.gamma_label.grid(column=0, row=6, sticky='w')
        self.gamma_input = ttk.Entry(self)
        self.gamma_input.grid(column=1, row=6)
        self.gamma_input.insert(0, 'auto')
        self.is_checked = IntVar()
        self.is_balanced_checkbutton = ttk.Checkbutton(self, text='设置为平衡权重', variable=self.is_checked)
        self.is_balanced_checkbutton.grid(column=2, row=6)
        self.is_checked.set(1)

        self.C_label = ttk.Label(self, text='C=')
        self.C_label.grid(column=0, row=7, sticky='w')
        self.C_input = ttk.Entry(self)
        self.C_input.grid(column=1, row=7)
        self.C_input.insert(0, '10')
        self.train_button = ttk.Button(self, text='训练模型', command=self.ui_train_svm)
        self.train_button.grid(column=2, row=7, sticky='e')

        self.plot_button = ttk.Button(self, text='显示ROC曲线', command=self.ui_plot_ROC)
        self.plot_button.grid(column=1, row=8)

        self.pie_button = ttk.Button(self, text='显示饼图', command=self.pie_chart)
        self.pie_button.grid(column=0, row=8, sticky='w')

        self.scatter_button = ttk.Button(self, text='显示散点图', command=self.plot_scatter)
        self.scatter_button.grid(column=2, row=8, sticky='e')

        ttk.Separator(self, orient='horizontal').grid(column=0,
                row=1, columnspan=4, sticky='ew')

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=10)

    def get_three_filename(self):
        def filename_split(filename):
            return os.path.split(filename)[1]

        train_file = filename_split(self.input_train_file.get()) or 'ChnSentiCorp_htl_unba_10000'
        test_file = filename_split(self.input_test_file.get()) or 'ChnSentiCorp_test'
        model_name = filename_split(self.input_model_name.get()) or 'ChnSentiCorp_svc.pkl'
        return train_file, test_file, model_name

    def pie_chart(self):
        train_file, test_file, model_name = self.get_three_filename()
        visual.use_pie_chart(train_file=train_file, test_file=test_file, svc_model_name=model_name)

    def plot_scatter(self):
        train_file, test_file, model_name = self.get_three_filename()
        tnt.plot_scatter(train_file=train_file, test_file=test_file, svc_model_name=model_name)

    def select_train_file(self):
        self.input_train_file.delete(0, END)
        file_path = filedialog.askdirectory()
        if file_path:
            self.input_train_file.insert(0, file_path)

    def select_test_file(self):
        self.input_test_file.delete(0, END)
        file_path = filedialog.askdirectory()
        if file_path:
            self.input_test_file.insert(0, file_path)

    def select_model_file(self):
        self.input_model_name.delete(0, END)
        file_path = filedialog.askopenfilename()
        if file_path:
            self.input_model_name.insert(0, file_path)

    def get_param(self):
        C = self.C_input.get()
        gamma = self.gamma_input.get()
        is_balanced = self.is_checked.get()
        if gamma != 'auto':
            gamma = float(gamma)
        return int(C), int(is_balanced), gamma

    def ui_train_svm(self):
        train_file, test_file, model_name = self.get_three_filename()
        C, is_balanced, gamma = self.get_param()
        tnt.train_svm(train_file=train_file, svc_model_name=model_name, C=C, is_balanced=is_balanced, gamma=gamma)

    def ui_plot_ROC(self):
        train_file, test_file, model_name = self.get_three_filename()
        C, is_balanced, gamma = self.get_param()
        tnt.plot_ROC(train_file, test_file, model_name, C=C, is_balanced=is_balanced, gamma=gamma)

    def qs_hotel(self):
        self.input_train_file.delete(0, END)
        self.input_test_file.delete(0, END)
        self.input_model_name.delete(0, END)

        self.input_train_file.insert(0, 'ChnSentiCorp_htl_unba_10000')
        self.input_test_file.insert(0, 'ChnSentiCorp_test2')
        self.input_model_name.insert(0, 'ChnSentiCorp_svc.pkl')


    def qs_movie(self):
        self.input_train_file.delete(0, END)
        self.input_test_file.delete(0, END)
        self.input_model_name.delete(0, END)

        self.input_train_file.insert(0, 'xiyou_movie')
        self.input_test_file.insert(0, 'test_xiyou_movie')
        self.input_model_name.insert(0, 'xiyou_movie_svc.pkl')

    def qs_movie_all(self):
        self.input_train_file.delete(0, END)
        self.input_test_file.delete(0, END)
        self.input_model_name.delete(0, END)

        self.input_train_file.insert(0, 'xiyou_movie_all')
        self.input_test_file.insert(0, 'test_xiyou_movie')
        self.input_model_name.insert(0, 'all_xiyou_movie_svc.pkl')

    def on_quit(self):
        quit()

if __name__ == '__main__':
    mainFrame = tkinter.Tk()
    AddWidget(mainFrame)
    mainFrame.mainloop()
