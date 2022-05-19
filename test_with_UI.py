# -*- coding: utf-8 -*
import os
import threading
import time

import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import db
import ui_util

from login import Ui_MainWindow as loginw
from menu import Ui_MainWindow as menuw
from trainui import Ui_MainWindow as trainw
import trainconfig as args

import os
import sys
import random
from PIL import Image

import glob
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from models import TransformerNet, VGG16
from utils import *


class Ui_MainWindow(QWidget):
    # 界面主体布局
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1255, 790)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 30, 1221, 691))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label.setMinimumSize(QtCore.QSize(400, 300))
        self.label.setMaximumSize(QtCore.QSize(400, 300))
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.label.setText("")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget)     # 显示图片的框
        self.label_3.setMinimumSize(QtCore.QSize(400, 0))
        self.label_3.setMaximumSize(QtCore.QSize(400, 300))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)
        self.btn3 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn3.setMaximumSize(QtCore.QSize(150, 40))
        self.btn3.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.btn3.setObjectName("btn3")
        self.gridLayout.addWidget(self.btn3, 0, 2, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.gridLayoutWidget)  # 下拉列表
        self.comboBox.setMaximumSize(QtCore.QSize(240, 40))
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_2.setMinimumSize(QtCore.QSize(400, 0))
        self.label_2.setMaximumSize(QtCore.QSize(400, 300))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.btn1 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.btn1.setMaximumSize(QtCore.QSize(150, 40))
        self.btn1.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))
        self.btn1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn1.setObjectName("btn1")
        self.gridLayout.addWidget(self.btn1, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_4.setMaximumSize(QtCore.QSize(160, 40))
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setMaximumSize(QtCore.QSize(160, 40))
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setMaximumSize(QtCore.QSize(160, 40))
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1255, 30))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # 界面优化，控件属性设置，动作连接
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn3.setText(_translate("MainWindow", "运行"))
        self.btn1.setText(_translate("MainWindow", "选择图片"))
        self.label_4.setText(_translate("MainWindow", "原始图片"))
        self.label_5.setText(_translate("MainWindow", "风格图片"))
        self.label_6.setText(_translate("MainWindow", "输出图片"))
        self.label.setStyleSheet("border: 2px solid red")
        self.label.setScaledContents(True)
        self.label_2.setStyleSheet("border: 2px solid red")
        self.label_2.setScaledContents(True)
        self.label_3.setStyleSheet("border: 2px solid red")
        self.label_3.setScaledContents(True)
        self.btn1.clicked.connect(self.setpath)
        self.initstyles()
        self.comboBox.currentIndexChanged.connect(self.setstyle)
        self.btn3.clicked.connect(self.showresult)

    # 设置路径
    def setpath(self):
        filepath, filetype = QFileDialog.getOpenFileName(
            self, '选择文件', '', '(*.jpg *.png)')
        ui_util.input_path = filepath
        pic = QPixmap(ui_util.input_path)
        self.label.setPixmap(pic)

    # 初始化下拉列表
    def initstyles(self):
        temp = ui_util.model2pic
        for item in temp:
            self.comboBox.addItem(item)

    # 显示风格图片
    def setstyle(self):
        name = self.comboBox.currentText()
        pic = QPixmap(ui_util.model2pic[name])
        print(ui_util.model2pic[name])
        ui_util.model_path = ui_util.model2path[name]
        self.label_2.setPixmap(pic)

    # 显示算法输出结果
    def showresult(self):
        ui_util.run()
        pic = QPixmap(ui_util.output_path)
        self.label_3.setPixmap(pic)

class loginWindow(QMainWindow,loginw):
    def __init__(self):
        super(loginWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("系统登录")
        self.setMinimumSize(435, 328)
        self.setMaximumSize(435, 328)
        #事件绑定
        self.pushButton.clicked.connect(self.login)
        self.pushButton_2.clicked.connect(self.reg)
        self.action.triggered.connect(self.reg)
        self.action_2.triggered.connect(self.about)
        self.isr = False
    #登录
    def login(self):
       if self.lineEdit.text() and self.lineEdit_2.text():
            res = db.findBy("select username from user where username = %s",(self.lineEdit.text()))
            if (len(res) <= 0):
                QMessageBox.warning(self, "提示", '该用户不存在', QMessageBox.Ok)
                return
            res = db.findBy("select username from user where username = %s and pwd = %s",(self.lineEdit.text(),self.lineEdit_2.text()))
            if(len(res)>0):
                QMessageBox.warning(self, "提示", '登录成功', QMessageBox.Ok)
                menu.show()
                self.close()
            else:
                QMessageBox.warning(self, "提示", '账号密码错误', QMessageBox.Ok)
       else:
           QMessageBox.warning(self, "提示", '请输入账号或密码！', QMessageBox.Ok)
    #注册
    def reg(self):
        if self.lineEdit.text() and self.lineEdit_2.text():
            sql = "insert user(username,pwd) values(%s,%s)"
            if db.insert(sql,(self.lineEdit.text(),self.lineEdit_2.text())):
                QMessageBox.warning(self, "提示", '注册成功', QMessageBox.Ok)
                self.isr = True
            else:
                QMessageBox.warning(self, "提示", '注册失败', QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "提示", '请输入账号或密码！', QMessageBox.Ok)
    #关于
    def about(self):
        QMessageBox.about(self, "提示", '如需帮助，请与我联系，联系方式QQxxxxx')


class mennuWindow(QMainWindow,menuw):
    def __init__(self):
        super(mennuWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("主界面")
        self.setMinimumSize(677, 447)
        self.setMaximumSize(677, 447)
        self.pushButton.clicked.connect(self.train)
        self.pushButton_2.clicked.connect(self.pre)
        self.isr = False
    #打开训练界面
    def train(self):
        train.show()
        self.close()
    #打开图像迁移界面
    def pre(self):
        #self.close()
        ui.show()

class trainWindow(QMainWindow,trainw):
    conn = pyqtSignal(str)
    def __init__(self):
        super(trainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("训练模型")
        self.setMinimumSize(733, 600)
        self.setMaximumSize(733, 600)
        self.pushButton.clicked.connect(self.btn1)
        self.pushButton_2.clicked.connect(self.btn2)
        #self.pushButton_3.clicked.connect(self.btn3)
        self.pushButton_4.clicked.connect(self.train)
        self.pushButton_5.clicked.connect(self.back)
        self.conn.connect(self.handle_str)
        self.isr = False
    def handle_str(self,s):
        print(s)
        if s =="训练成功":
            QMessageBox.warning(self, "提示", '训练成功', QMessageBox.Ok)
            self.pushButton_4.setText("开始训练")
            self.pushButton_4.setEnabled(True)
        if s =="训练失败":
            QMessageBox.warning(self, "提示", '训练失败', QMessageBox.Ok)
            self.pushButton_4.setText("开始训练")
            self.pushButton_4.setEnabled(True)
        # if s !="训练成功"and s!="训练失败":
        #     self.lineEdit_13.setText(s)
    #选取数据集路径
    def btn1(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(None, "请选择文件夹路径", os.getcwd())
        self.lineEdit.setText(directory)
        args.dataset_path = directory
    #选取样式文件
    def btn2(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(None, "选取文件", os.getcwd(),
                                                                   "All Files(*);;Text Files(*.txt)")
        args.style_image=fileName
        self.lineEdit_2.setText(fileName)
     #选择输出文件路径
    # def btn3(self):
    #     directory = QtWidgets.QFileDialog.getExistingDirectory(None, "请选择文件夹路径", "D:/")
    #     self.lineEdit_3.setText(directory)

    #训练模型的函数
    def train(self):
        try:
            self.pushButton_4.setText("正在训练")
            self.pushButton_4.setEnabled(False)
            args.epochs = int(self.lineEdit_4.text())
            args.batch_size = int(self.lineEdit_5.text())
            args.image_size = int(self.lineEdit_6.text())
            args.style_size = int(self.lineEdit_7.text())
            args.lambda_content = float(self.lineEdit_8.text())
            args.lambda_style= float(self.lineEdit_9.text())
            args.lr = float(self.lineEdit_10.text())
            args.checkpoint_interval = int(self.lineEdit_11.text())
            args.sample_interval = int(self.lineEdit_12.text())
            th = threading.Thread(target=self.handle)
            th.start()
        except Exception as e:
            QMessageBox.warning(self, "提示", '参数不正确，无法开始训练', QMessageBox.Ok)
            self.pushButton_4.setText("开始训练")
            self.pushButton_4.setEnabled(True)
            print(e)


    #返回主界面
    def back(self):
        menu.show()
        self.close()

    def handle(self):
        try:
            style_name = args.style_image.split("\\")[-1].split(".")[0].split("/")[-1]
            print(style_name)
            os.makedirs(f"images/outputs/{style_name}-training", exist_ok=True)
            os.makedirs(f"checkpoints", exist_ok=True)
            # 输出文件夹，如果不存在则创建之
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(args.dataset_path)
            # 如果电脑有编译好CUDA的GPU，则使用之。如果没有，就用CPU运行这个项目。
            train_dataset = datasets.ImageFolder(args.dataset_path, train_transform(args.image_size))
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
            # 这里是定义dataloader，说白了就是定义训练过程中读取数据集的方法和速度。
            # 关于dataloader是干什么的参考https://pytorch.org/docs/stable/data.html。

            # 定义模型，这段代码是用Transformer实现的模型，模型需要在使用前实例化，然后传送到GPU（或者CPU，同上如果你电脑有GPU的话）。
            transformer = TransformerNet().to(device)
            # Transformer前用VGG16提取特征。所以也要初始化VGG模型，所以这里是不需要记录VGG的梯度的，只需要用它推理就好了。
            vgg = VGG16(requires_grad=False).to(device)
            # Load checkpoint model if specified
            # 从本地的训练好的模型中，加载模型的权重。
            if args.checkpoint_model:
                transformer.load_state_dict(torch.load(args.checkpoint_model))
            # 定义 优化器 和损失函数，这里用了Adam优化器和MSELoss，关于MSE的理解:https://blog.csdn.net/weixin_38145317/article/details/103735784
            optimizer = Adam(transformer.parameters(), args.lr)  # 优化器
            l2_loss = torch.nn.MSELoss().to(device)  # 损失函数
            # 用PIL库把本地的目标风格图片读进来，预处理归一化，转成tensor，送到GPU里（或者CPU，同上）
            style = style_transform(args.style_size)(Image.open(args.style_image))
            style = style.repeat(args.batch_size, 1, 1, 1).to(device)
            # 把上面风格图片的特征提出来，就是我们想学习的特征。
            features_style = vgg(style)
            gram_style = [gram_matrix(y) for y in features_style]

            # 随机提取 8 幅图像用于模型的视觉评估，通俗来讲就是看看效果怎么样
            image_samples = []
            for path in random.sample(glob.glob(f"{args.dataset_path}/*/*.jpg"), 8):
                # image_samples += [style_transform(args.image_size)(Image.open(path))]
                image_samples += [train_transform(args.image_size)(Image.open(path))]
            image_samples = torch.stack(image_samples)
            # 把随机提取的 8 幅图像，做一下视觉迁移，保存，看看效果。
            def save_sample(batches_done):
                """ Evaluates the model and saves image samples """
                transformer.eval()
                with torch.no_grad():
                    output = transformer(image_samples.to(device))
                image_grid = denormalize(torch.cat((image_samples.cpu(), output.cpu()), 2))
                save_image(image_grid, f"images/outputs/{style_name}-training/{batches_done}.jpg", nrow=4)
                transformer.train()

            # 对每个epoch
            for epoch in range(args.epochs):
                # 这里是定义了一些metrics，实际上是风格迁移从各个方向定义的loss，定义了视觉迁移的效果。
                epoch_metrics = {"content": [], "style": [], "total": []}
                for batch_i, (images, _) in enumerate(dataloader):
                    # 从dataloader中按batch抽取图像数据集。
                    # 在训练期间将梯度定义为0，关于为什么这么做：
                    # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                    optimizer.zero_grad()

                    # 显卡和CPU是不能互访的，所以既然你把模型传到了其中某个里面，所有的数据也都要传过去。
                    images_original = images.to(device)
                    # 获取transform之后的图像
                    images_transformed = transformer(images_original)

                    # 用VGG提取原图和transform之后的特征，用来计算loss。这一步是用来防止图像“过于”风格化
                    features_original = vgg(images_original)
                    features_transformed = vgg(images_transformed)

                    # Compute content loss as MSE between features
                    content_loss = args.lambda_content * l2_loss(features_transformed.relu2_2, features_original.relu2_2)

                    # 计算transformer模型输出的图像和风格图像的风格特征是否相近的loss
                    style_loss = 0
                    for ft_y, gm_s in zip(features_transformed, gram_style):  # (1,2,3) (4,5,6)  -> zip -> (1,4)(2,5)(3,6)
                        gm_y = gram_matrix(ft_y)
                        style_loss += l2_loss(gm_y, gm_s[: images.size(0), :, :])
                    style_loss *= args.lambda_style

                    # 这俩loss加起来，作为总共的loss（既关注，又关注）。
                    total_loss = content_loss + style_loss

                    # 依照loss进行方向传播
                    total_loss.backward()
                    # 学习率也要随着step下降
                    optimizer.step()

                    # 下面没什么好说的，把loss打印到屏幕上，规范化输出之类的。
                    epoch_metrics["content"] += [content_loss.item()]
                    epoch_metrics["style"] += [style_loss.item()]
                    epoch_metrics["total"] += [total_loss.item()]
                    strr = "\r[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]"% (epoch + 1,
                            args.epochs,
                            batch_i,
                            (len(train_dataset) / args.batch_size),
                            content_loss.item(),
                            np.mean(epoch_metrics["content"]),
                            style_loss.item(),
                            np.mean(epoch_metrics["style"]),
                            total_loss.item(),
                            np.mean(epoch_metrics["total"]),
                        )
                    sys.stdout.write(
                        strr
                    )
                    #self.conn.emit(strr)

                    batches_done = epoch * len(dataloader) + batch_i + 1
                    # 每隔几步看一下效果咋样（对sample的8张图推理看看效果）
                    if batches_done % args.sample_interval == 0:
                        save_sample(batches_done)

                    # 每隔几步保存一下checkpoint
                    if args.checkpoint_interval > 0 and batches_done % args.checkpoint_interval == 0:
                        style_name = os.path.basename(args.style_image).split(".")[0]
                        torch.save(transformer.state_dict(), f"checkpoints/{style_name}_{batches_done}.pth")

            self.conn.emit("训练成功")

        except Exception as e:
            self.conn.emit("训练失败")
            print(e)
            # self.pushButton_4.setText("开始训练")
            # self.pushButton_4.setEnabled(True)
            # QMessageBox.warning(self, "提示", '训练失败', QMessageBox.Ok)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = QMainWindow()
    wds = Ui_MainWindow()
    wds.setupUi(ui)
    ui.setWindowTitle("图像迁移")
    ui_help = QMainWindow()
    login = loginWindow()#登录界面
    menu = mennuWindow()#主面板界面
    train = trainWindow()#训练模型界面
    login.show()#展示登录界面
    #train.show()测试用

    sys.exit(app.exec_())
