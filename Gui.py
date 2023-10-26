import tkinter as tk
from tkinter import *
from tkinter import messagebox
from model_5 import PoetryModellstm5
from model_7 import PoetryModellstm7
import torch
import re

# 窗口大小
page_size = '800x500+900+450'

#
params = {
    "batch_size": 32,  # batch(古诗数)大小
    "epochs": 1000,  # epoch大小
    "lr": 0.003,  # 学习率
    "hidden_num": 64,  # 隐层大小
    "embedding_num": 128,  # 词向量维度
    "train_num": 2000,  # 训练的古诗数量, 七言古诗:0~6290, 五言古诗:0~2929
    "optimizer": torch.optim.AdamW,  # 优化器 , 注意不要加括号
    "batch_num_test": 100,  # 多少个batch 打印一首古诗进行效果测试
}


# 主页面
class MainPage(object):
    # 初始化页面
    def __init__(self, master_page):
        self.page = None
        self.root = master_page
        self.root.geometry(page_size)
        self.init_page()

    def init_page(self):
        self.page = tk.Frame(self.root)
        self.page.pack(fill='both', ipadx=15, ipady=10, expand=True)

        '''
        '具体布局
        '''
        # 标题
        title_label = tk.Label(self.page, text='请选择诗的格式', height=3, width=200, bg='white',
                               font=('Arial', 14))
        title_label.pack()

        # 设计按钮的样式，大小和位置
        # 跳转五言诗
        to_five_button = tk.Button(self.page, text='    五言    ',
                                   command=self.to_five_chars, width=15, height=3)
        to_five_button.place(relx=0.425, rely=0.3)
        # 跳转七言诗
        to_seven_button = tk.Button(self.page, text='    七言    ', command=self.to_seven_chars,  width=15, height=3)
        to_seven_button.place(relx=0.425, rely=0.6)

    # 跳转
    def to_five_chars(self):
        self.page.destroy()
        FiveChars(self.root)

    def to_seven_chars(self):
        self.page.destroy()
        SevenChars(self.root)


# 五言诗页面
class FiveChars(object):
    # 初始化页面
    def __init__(self, master_page):
        self.page = None
        self.root = master_page
        self.acrostic = tk.StringVar()
        self.var = IntVar()
        self.root.geometry(page_size)
        self.init_page()

    def init_page(self):
        self.page = tk.Frame(self.root)
        self.page.pack(fill='both', ipadx=15, ipady=10, expand=True)

        '''
        '功能实现
        '''

        # 切换模式使藏头输入框消失和出现
        def toggle_input():
            if self.var.get() == 0:
                acrostic_input.place_forget()
            else:
                acrostic_input.place(relx=0.4, rely=0.3)

        # 实现输入框默认提示
        def input_fun(event):
            if self.acrostic.get() == '请输入藏头':
                acrostic_input.delete('0', 'end')
            elif self.acrostic.get() == '':
                acrostic_input.insert('insert', '请输入藏头')


        # 生成诗歌
        def get_poetry():
            if self.var.get() == 0:
                model = PoetryModellstm5(params)  # 模型定义
                model = model.to_train()
                key = model.generate_poetry_auto()
                print(key)
            else:
                if self.acrostic.get() == '':
                    messagebox.showerror('err', '请输入藏头')
                    return -1
                if len(self.acrostic.get()) < 4:
                    messagebox.showerror('err', '藏头字数必须大于等于4')
                    return -1
                if len(self.acrostic.get()) % 2 == 0:
                    for ch in self.acrostic.get():
                        if ch < u'\u4e00' or ch > u'\u9fff':
                            messagebox.showerror('err', '输入含有非汉字字符,请重新输入')
                            return -1
                    model = PoetryModellstm5(params)  # 模型定义
                    model = model.to_train()
                    key = model.generate_poetry_acrostic(self.acrostic.get())
                else:
                    messagebox.showerror('err', '请输入偶数个藏头字')
                    return -1
            poetry_show.delete(0.0, tk.END)
            keys = re.split('，|\。', key)
            i = 0
            while True:
                if keys[i] == '':
                    break
                else:
                    poetry_show.insert('insert', keys[i] + '，' + keys[i + 1] + '。' + '\n')
                    i += 2

        '''
        '具体布局
        '''
        # 标题
        title_label = tk.Label(self.page, text='获取五言诗', height=3, width=200, bg='white',
                               font=('Arial', 14))
        title_label.pack()

        # 藏头输入
        acrostic_input = tk.Entry(self.page, textvariable=self.acrostic)
        acrostic_input.insert('insert', '请输入藏头')
        acrostic_input.bind("<FocusIn>", input_fun)
        acrostic_input.bind("<FocusOut>", input_fun)
        acrostic_input.place(relx=0.4, rely=0.3)

        # 诗歌生成框
        poetry_show = Text(self.page, height=8, width=25)
        poetry_show.place(relx=0.38, rely=0.4)

        self.var.set(1)
        btn_randon = tk.Radiobutton(self.page, text='随机生成', variable=self.var, value=0, command=toggle_input)
        btn_randon.place(relx=0.35, rely=0.2)
        btn_acrostic = tk.Radiobutton(self.page, text='藏头诗', variable=self.var, value=1, command=toggle_input)
        btn_acrostic.place(relx=0.55, rely=0.2)

        # 诗歌获取按钮
        get_poetry_button = tk.Button(self.page, text='生成诗歌',
                                      command=get_poetry)
        get_poetry_button.place(relx=0.35, rely=0.7)

        # 返回按钮
        back_button = tk.Button(self.page, text='返回',
                                command=self.to_back)
        back_button.place(relx=0.6, rely=0.7)

    # 跳转
    def to_back(self):
        self.page.destroy()
        MainPage(self.root)


# 七言诗页面
class SevenChars(object):
    # 初始化页面
    def __init__(self, master_page):
        self.page = None
        self.root = master_page
        self.acrostic = tk.StringVar()
        self.var = IntVar()
        self.root.geometry(page_size)
        self.init_page()

    def init_page(self):
        self.page = tk.Frame(self.root)
        self.page.pack(fill='both', ipadx=15, ipady=10, expand=True)

        '''
        '功能实现
        '''

        # 切换模式使藏头输入框消失和出现
        def toggle_input():
            if self.var.get() == 0:
                acrostic_input.place_forget()
            else:
                acrostic_input.place(relx=0.4, rely=0.3)

        # 实现输入框默认提示
        def input_fun(event):
            if self.acrostic.get() == '请输入藏头':
                acrostic_input.delete('0', 'end')
            elif self.acrostic.get() == '':
                acrostic_input.insert('insert', '请输入藏头')


        # 生成诗歌
        def get_poetry():
            poetry_show.delete(0.0, tk.END)
            if self.var.get() == 0:
                model = PoetryModellstm7(params)  # 模型定义
                model = model.to_train()
                key = model.generate_poetry_auto()
            else:
                if self.acrostic.get() == '':
                    messagebox.showerror('err', '请输入藏头')
                    return -1
                if len(self.acrostic.get()) < 4:
                    messagebox.showerror('err', '藏头字数必须大于等于4')
                    return -1
                if len(self.acrostic.get()) % 2 == 0:
                    for ch in self.acrostic.get():
                        if ch < u'\u4e00' or ch > u'\u9fff':
                            messagebox.showerror('err', '输入含有非汉字字符,请重新输入')
                            return -1
                    model = PoetryModellstm7(params)  # 模型定义
                    model = model.to_train()
                    key = model.generate_poetry_acrostic(self.acrostic.get())
                else:
                    messagebox.showerror('err', '请输入偶数个藏头字')
                    return -1
            keys = re.split('，|\。', key)
            i = 0
            while True:
                if keys[i] == '':
                    break
                else:
                    poetry_show.insert('insert', keys[i] + '，' + keys[i + 1] + '。' + '\n')
                    i += 2

        '''
        '具体布局
        '''
        # 标题
        title_label = tk.Label(self.page, text='获取七言诗', height=3, width=200, bg='white',
                               font=('Arial', 14))
        title_label.pack()

        # 藏头输入
        acrostic_input = tk.Entry(self.page, textvariable=self.acrostic)
        acrostic_input.insert('insert', '请输入藏头')
        acrostic_input.bind("<FocusIn>", input_fun)
        acrostic_input.bind("<FocusOut>", input_fun)
        acrostic_input.place(relx=0.4, rely=0.3)

        # 诗歌生成框
        poetry_show = Text(self.page, height=8, width=35)
        poetry_show.place(relx=0.35, rely=0.4)

        self.var.set(1)
        btn_randon = tk.Radiobutton(self.page, text='随机生成', variable=self.var, value=0, command=toggle_input)
        btn_randon.place(relx=0.35, rely=0.2)
        btn_acrostic = tk.Radiobutton(self.page, text='藏头诗', variable=self.var, value=1, command=toggle_input)
        btn_acrostic.place(relx=0.55, rely=0.2)

        # 诗歌获取按钮
        get_poetry_button = tk.Button(self.page, text='生成诗歌',
                                      command=get_poetry)
        get_poetry_button.place(relx=0.35, rely=0.7)

        # 返回按钮
        back_button = tk.Button(self.page, text='返回',
                                command=self.to_back)
        back_button.place(relx=0.6, rely=0.7)

    # 跳转
    def to_back(self):
        self.page.destroy()
        MainPage(self.root)


page = tk.Tk()
page.geometry('600x400+900+450')
# 窗口名称
page.title('AI写诗系统')
# 禁止调节窗口大小
page.resizable(False, False)
MainPage(page)
page.mainloop()
