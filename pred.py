import tensorflow as tf
import numpy as np
from cnn_train import load_data
import matplotlib.pyplot as plt


def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()

    im = ax.imshow(harvest, cmap="Blues")

    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))

    # 添加每个热力块的具体数值
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, round(harvest[i, j], 2),
                           ha="center", va="center", color="black")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)
    # plt.show()


def predict(model_name):

    train_dataset, test_dataset, class_names = load_data()

    # 加载模型
    model = tf.keras.models.load_model("models/" + model_name + ".h5")
    model.summary()

    # 测试
    loss, accuracy = model.evaluate(test_dataset)

    # 输出结果
    print('test accuracy :', accuracy)

    test_real_labels = []  # 真实结果类名
    test_pre_labels = []   # 测试结果类名
    for test_batch_images, test_batch_labels in test_dataset:
        test_batch_labels = test_batch_labels.numpy()
        test_batch_pres = model.predict(test_batch_images)
        # print(test_batch_pres)

        test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
        test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
        # print(test_batch_labels_max)
        # print(test_batch_pres_max)
        # 将推理对应的标签取出
        for i in test_batch_labels_max:
            test_real_labels.append(i)

        for i in test_batch_pres_max:
            test_pre_labels.append(i)
        # break

    # print(test_real_labels)
    # print(test_pre_labels)
    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

    print(heat_maps)
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    # print(heat_maps_sum)
    print()
    heat_maps_float = heat_maps / heat_maps_sum
    print(heat_maps_float)
    # title, x_labels, y_labels, harvest
    show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                  save_name="results/heatmap_"+ model_name +".png")


if __name__ == '__main__':
    predict("monkey_cnn")
    predict("ResNet152V2")