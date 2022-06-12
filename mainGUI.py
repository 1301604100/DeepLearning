# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 20:29
# @Author  : dejahu
# @Email   : 1148392984@qq.com
# @File    : window.py
# @Software: PyCharm
# @Brief   : 图形化界面

import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil


class MainWindow(QTabWidget):
    # 初始化
    def __init__(self):
        super().__init__()
        # 系统名称
        self.setWindowTitle('猴子识别')
        # 模型初始化
        self.model = tf.keras.models.load_model("models/ResNet152V2.h5")

        # 类名
        self.chinese_class_names = ['赤秃猴', '黑夜猴', '松鼠猴', '日本猕猴', '鬃毛吼猴', '尼尔吉里叶猴', '红猴', '侏儒狨猴', '银毛猴', '白头卷尾猴']
        self.english_class_names = ['bald_uakari', 'black_headed_night_monkey', 'common_squirrel_monkey', 'japanese_macaque',
                   'mantled_howler', 'nilgiri_langur', 'patas_monkey', 'pygmy_marmoset', 'silvery_marmoset',
                   'white_headed_capuchin']
        self.latin_class_names = ['alouatta_palliata', 'erythrocebus_patas', 'cacajao_calvus', 'macaca_fuscata', 'cebuella_pygmea',
                                  'cebus_capucinus', 'mico_argentatus', 'saimiri_sciureus', 'aotus_nigriceps', 'trachypithecus_johnii']

        self.init_image = "D:/pycode/monkey_tf/images/init_image.jpg"
        self.resize(900, 700)
        self.initUI()

    # 界面初始化，设置界面布局
    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('楷体', 20)

        # 主页面，设置组件并在组件放在布局上
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        # 标题
        img_title = QLabel("待识别图片")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)

        # 要识别的图
        self.img_label = QLabel()
        img_init = cv2.imread(self.init_image)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images/show.png", img_show)
        img_init = cv2.resize(img_init, (300, 300))
        cv2.imwrite('images/target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))

        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_layout = QVBoxLayout()

        # 上传
        btn_upload = QPushButton(" 上传图片 ")
        btn_upload.clicked.connect(self.upload_img)
        btn_upload.setFont(font)
        btn_upload.setStyleSheet('''QPushButton{background:skyblue;border-radius:5px;}''')

        # 识别
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)
        btn_predict.setStyleSheet('''QPushButton{background:skyblue;border-radius:5px;}''')

        # 结果
        label_result = QLabel(' 猴子名称 ')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('楷体', 20))
        self.result.setFont(QFont('楷体', 24))

        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addStretch()
        right_layout.addWidget(btn_upload)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)

        self.addTab(main_widget, '主页')

    # 上传并显示图片
    def upload_img(self):
        # 打开文件选择框选择文件
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'Image files(*.jpg *.png *jpeg)')
        # 获取图片名称
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:

            target_image_name = "images/upload_image." + img_name.split(".")[-1]
            # 将图片移动到当前目录
            shutil.copy(img_name, target_image_name)
            # 打开图片
            img_init = cv2.imread(target_image_name)
            h, w, c = img_init.shape
            scale = 400 / h
            # 将图片的大小统一调整到400的高，方便界面显示
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("images/show.png", img_show)
            # 将图片大小调整到 300 * 300 用于模型推理
            img_init = cv2.resize(img_init, (300, 300))
            cv2.imwrite('images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("images/show.png"))
            self.result.setText("等待识别")

    # 预测图片
    def predict_img(self):
        # 读取图片
        img = Image.open('images/target.png')
        # 将图片转化为numpy的数组
        img = np.asarray(img)
        # 将图片输入模型得到结果
        outputs = self.model.predict(img.reshape(1, 300, 300, 3))
        result_index = int(np.argmax(outputs))
        # 获得对应的猴子名
        chinese_result = self.chinese_class_names[result_index]
        english_result = self.english_class_names[result_index]
        latin_result = self.latin_class_names[result_index]
        # 在界面上做显示
        self.result.setText("中文名: " + chinese_result +
                            "\n英文名: " + english_result +
                            "\n拉丁名: " + latin_result)

    # 界面关闭事件，询问用户是否关闭
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
