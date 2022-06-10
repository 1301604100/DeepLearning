import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from cnn_train import load_data


def predict(model_name):
    def preprocess(x, y):  # 预处理代码
        x = tf.cast(x, dtype=tf.float32) / 255.
        return x, y

    np.random.seed(2021)  # 固定随机因子
    __, test_db, _ = load_data()

    test_db = test_db.batch(1000)
    test_db = test_db.map(preprocess)  # 使用预处理程序
    model = keras.models.load_model('./models/' + model_name, compile=False)  # 加载模型
    model.summary()  # 展现模型结构

    y_predict = model.predict(test_db)  # 预测测试集
    acc = keras.metrics.SparseCategoricalAccuracy()(y_test, y_predict)  # 计算准确率
    # y_test是(10000, 1), y_pred是(10000, 10)独热编码

    y_predict = tf.argmax(y_predict, 1)  # 转换为预测标签
    # (10000, )
    cm = confusion_matrix(y_test, y_predict)  # 混淆矩阵
    print(cm)  # 输出混淆矩阵
    # plotcm(cm)  # 绘制混淆矩阵, 自定义
