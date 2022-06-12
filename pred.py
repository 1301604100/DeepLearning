import tensorflow as tf
import numpy as np
from cnn_train import load_data
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


# 热力图
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
    flag = True

    for test_images, test_labels in test_dataset:
        test_labels = test_labels.numpy()  # 取出类名数组  0  1
        test_pres = model.predict(test_images)  # 进行预测

        # print(test_images[0])

        test_labels_max = np.argmax(test_labels, axis=1)  # 取出最大值所对应的索引
        test_pres_max = np.argmax(test_pres, axis=1)  # 取出最大值所对应的索引

        if flag:
            # 找出错误分类
            flag = draw_mis_classification(test_labels_max, test_pres_max, test_images, model_name)

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

    # 计算评价指标
    evaluating_indicator(cm, class_names_length, n)

    print()
    # 获得概率的混淆矩阵
    cm_float = cm / cm_sum
    print(cm_float)

    # 热力图
    show_heatmaps(x_labels=class_names,
                  y_labels=class_names,
                  cm=cm_float,
                  save_name="results/heatmap_" + model_name + ".png")


# 计算评价指标
def evaluating_indicator(cm, length, n):
    # cm      混淆矩阵
    # length  分类数量
    # n       测试集数

    # https://blog.csdn.net/qwe1110/article/details/103391632
    t = p = r = 0
    for i in range(length):
        temp_p1 = temp_p2 = temp_r = 0
        for j in range(length):
            if i == j:
                t += cm[i][j]
                temp_p2 = cm[i][j]
            temp_p1 += cm[j][i]
            temp_r += cm[i][j]
        p += (temp_p2 / temp_p1)
        r += (temp_p2 / temp_r)
    print("准确度acc:", t / n)
    print("精确度p:", p / length)
    print("召回率r:", r / length)


# 画出错分样例图
def draw_mis_classification(test_labels_max, test_pres_max, test_images, photo_name):
    # classes = ['bald_uakari', 'black_headed_night_monkey', 'common_squirrel_monkey', 'japanese_macaque',
    #            'mantled_howler', 'nilgiri_langur', 'patas_monkey', 'pygmy_marmoset', 'silvery_marmoset',
    #            'white_headed_capuchin']

    classes = ['赤秃猴', '黑夜猴', '松鼠猴', '日本猕猴', '鬃毛吼猴', '尼尔吉里叶猴', '红猴', '侏儒狨猴', '银毛猴', '白头卷尾猴']

    img_label_true = []  # 正确的类名
    img_label_error = []  # 预测错误的类名
    img_error = []  # 预测错误的图片

    # 没找到错误的就继续
    if len(img_error) <= 0: return True

    # 找出错误分类
    for i, val in enumerate(test_labels_max):
        if test_labels_max[i] != test_pres_max[i]:
            img_label_true.append(test_labels_max[i])
            img_label_error.append(test_pres_max[i])
            img_error.append(test_images[i])

    # 显示的图片数
    show_img_count = 8
    error_label_len = len(img_label_error)
    num_for_paint = (error_label_len, show_img_count)[error_label_len > show_img_count]

    plt.figure()
    for i in range(num_for_paint):
        # 转换成数组
        numpy_out = np.array(img_error[i])

        plt.subplot(2, 4, i + 1, xticks=[], yticks=[])  # 2 * 4子图显示
        plt.imshow(numpy_out.astype(np.uint8))
        plt.title(f'{classes[img_label_true[i]]} --> \n {classes[img_label_error[i]]}')  # 显示标题
        plt.subplots_adjust(wspace=0.3, hspace=0.2)  # 调整子图间距
    plt.savefig('results/' + photo_name + '_error_label.png', dpi=100)
    # plt.show()
    return False


if __name__ == '__main__':
    predict("monkey_cnn")
    predict("ResNet152V2")
