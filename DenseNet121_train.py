import tensorflow as tf
from time import *
import matplotlib.pyplot as plt

# 指定数据位置
from draw import draw

base_path = "./data/monkey"
train_path = base_path + "/train"
validation_path = base_path + "/validation"

# 设置图片的宽高
image_width = 300
image_height = 300

# 一次训练所选取的样本数，迭代次数
batch_size = 32
epochs = 30


# 加载数据集，获取标签
def load_data():
    # 加载训练集
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,  # 数据所在目录
        label_mode='categorical',  # 标签被编码为分类向量
        seed=123,  # 用于shuffle和转换的可选随机种子
        image_size=(image_width, image_height),  # 重新调整大小
        batch_size=batch_size)  # 数据批次的大小
    # 加载测试集
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        validation_path,  # 数据所在目录
        label_mode='categorical',  # 标签被编码为分类向量
        seed=123,  # 用于shuffle和转换的可选随机种子
        image_size=(image_width, image_height),  # 重新调整大小
        batch_size=batch_size)  # 数据批次的大小
    # 获取分类名
    class_names = train_dataset.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_dataset, val_dataset, class_names


# 构建 CNN 模型
def get_model(class_num):

    # 使用官方的预训练模型
    base_model = tf.keras.applications.ResNet152V2(input_shape=(image_width, image_height, 3),   # 输入尺寸元组
                                                   include_top=False,     # 不包含顶层的全连接层
                                                   weights='imagenet')    # 表示使用官方预训练的权值

    # 将模型的主干参数进行冻结
    base_model.trainable = False
    model = tf.keras.models.Sequential([
        # 进行归一化的处理
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5,
                                                             offset=-1,
                                                             input_shape=(image_width, image_height, 3)),
        # 设置主干模型
        base_model,
        # 对主干模型的输出进行全局平均池化
        tf.keras.layers.GlobalAveragePooling2D(),
        # 通过全连接层映射到最后的分类数目上
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # 输出模型信息
    model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model


# 开始训练
def train():
    # 开始训练，记录开始时间
    begin_time = time()
    # 获取数据
    train_dataset, val_dataset, class_names = load_data()

    # 输出标签
    # print(class_names)

    # 构建 DenseNet121 模型
    model = get_model(len(class_names))
    # 开始训练
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    # 保存训练模型
    model.save("models/ResNet152V2.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print("程序运行时间: ", run_time, "s")

    # 绘制模型训练过程图
    draw(history, "ResNet152V2")


if __name__ == '__main__':
    train()
