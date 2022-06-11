import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from cnn_train import load_data
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


def show_heatmaps(x_labels, y_labels, cm, save_name):
    # 创建一个画布
    fig, ax = plt.subplots()

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))

    # 添加每个热力块的具体数值
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            ax.text(j, i, round(cm[i, j], 2), ha="center", va="center", color="black")
    # ax.set_xlabel("Predict label")
    # ax.set_ylabel("Actual label")

    plt.rc("font", family='Microsoft YaHei')

    ax.set_title("混淆矩阵")
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)


def predict(model_name):
    train_dataset, test_dataset, class_names = load_data()

    # 加载模型
    model = tf.keras.models.load_model("models/" + model_name + ".h5")
    model.summary()

    # 测试
    loss, accuracy = model.evaluate(test_dataset)

    # 输出结果
    print('测试准确度 :', accuracy)

    test_real_labels = []  # 真实结果类名
    test_pre_labels = []  # 测试结果类名
    for test_images, test_labels in test_dataset:
        test_labels = test_labels.numpy()  # 取出类名数组  0  1
        test_pres = model.predict(test_images)  # 进行预测

        test_labels_max = np.argmax(test_labels, axis=1)  # 取出最大值所对应的索引
        test_pres_max = np.argmax(test_pres, axis=1)  # 取出最大值所对应的索引

        # 取出真实结果的类名
        for i in test_labels_max:
            test_real_labels.append(i)
        # 取出预测结果的类名
        for i in test_pres_max:
            test_pre_labels.append(i)

    # 类名数量
    class_names_length = len(class_names)

    # 初始化混淆矩阵
    cm = np.zeros((class_names_length, class_names_length))
    # 构建混淆矩阵
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        cm[test_real_label][test_pre_label] += 1

    print(cm)

    # 每一行的总数
    cm_sum = np.sum(cm, axis=1).reshape(-1, 1)

    # 测试集数量
    n = len(test_real_labels)

    # https://blog.csdn.net/qwe1110/article/details/103391632
    t = p = r = 0
    for i in range(class_names_length):
        temp_p1 = temp_p2 = temp_r = 0
        for j in range(class_names_length):
            if i == j:
                t += cm[i][j]
                temp_p2 = cm[i][j]
            temp_p1 += cm[j][i]
            temp_r += cm[i][j]
        p += (temp_p2 / temp_p1)
        r += (temp_p2 / temp_r)
    print("准确度acc:", t / n)
    print("精确度p:", p / class_names_length)
    print("召回率r:", r / class_names_length)
    # print("f1:", 2 * p * r / (p + r))

    print()
    # 获得概率的混淆矩阵
    cm_float = cm / cm_sum
    print(cm_float)

    # 热力图
    show_heatmaps(x_labels=class_names,
                  y_labels=class_names,
                  cm=cm_float,
                  save_name="results/heatmap_" + model_name + ".png")


if __name__ == '__main__':
    predict("monkey_cnn")
    predict("ResNet152V2")
