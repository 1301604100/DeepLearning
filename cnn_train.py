import tensorflow as tf
from time import *
from draw import draw

# 指定数据位置
base_path = "./data/monkey"
train_path = base_path + "/train"
validation_path = base_path + "/validation"

# 设置图片的宽高
image_width = 300
image_height = 300

# 一次训练所选取的样本数，迭代次数
batch_size = 32
epochs = 30


# class_names = ['bald_uakari', 'black_headed_night_monkey', 'common_squirrel_monkey', 'japanese_macaque',
#                'mantled_howler', 'nilgiri_langur', 'patas_monkey', 'pygmy_marmoset', 'silvery_marmoset',
#                'white_headed_capuchin']


# 加载数据集，获取分类名
def load_data():
    # 加载训练集
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,  # 数据所在目录
        label_mode='categorical',  # 标签被编码为分类向量
        seed=2022,  # 用于shuffle和转换的可选随机种子
        image_size=(image_width, image_height),  # 重新调整大小
        batch_size=batch_size)  # 数据批次的大小
    # 加载测试集
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        validation_path,  # 数据所在目录
        label_mode='categorical',  # 标签被编码为分类向量
        seed=2022,  # 用于shuffle和转换的可选随机种子
        image_size=(image_width, image_height),  # 重新调整大小
        batch_size=batch_size)  # 数据批次的大小
    # 获取分类名
    class_names = train_dataset.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_dataset, val_dataset, class_names


# 卷积层（Convolutional layer），卷积神经网路中每层卷积层由若干卷积单元组成，每个卷积单元的参数都是通过反向传播算法优化得到的。
# 卷积运算的目的是提取输入的不同特征，第一层卷积层可能只能提取一些低级的特征如边缘、线条和角等层级，更多层的网络能从低级特征中迭代提取更复杂的特征。

# 线性整流层（Rectified Linear Units layer, ReLU layer），这一层神经的活性化函数（Activation function）
# 使用线性整流（Rectified Linear Units, ReLU）f(x)=max(0,x)f(x)=max(0,x)。

# 池化层（Pooling layer），通常在卷积层之后会得到维度很大的特征，将特征切成几个区域，取其最大值或平均值，得到新的、维度较小的特征。

# 全连接层（ Fully-Connected layer）, 把所有局部特征结合变成全局特征，用来计算最后每一类的得分。


# CNN 模型越深 就面临梯度弥散的问题，经过层数越多，则前面的信息就会渐渐减弱和消散
# 构建 CNN 模型
def get_model(class_num):
    # 搭建模型
    model = tf.keras.models.Sequential([
        # 对模型做归一化的处理，并表明输入图片的张量
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_width, image_height, 3)),

        # 卷积层1，该卷积层的输出为32个通道，卷积核的大小是3*3，激活函数为relu
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),   # ReLU解决梯度消失问题
        # 添加池化层1，池化的kernel大小是2*2
        tf.keras.layers.MaxPooling2D(2, 2),                      # 2 * 2 中选出最大值

        # 卷积层2，输出为64个通道，卷积核大小为3*3，激活函数为relu
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # 池化层2，最大池化，对2*2的区域进行池化操作
        tf.keras.layers.MaxPooling2D(2, 2),

        # 卷积层3，输出为128个通道，卷积核大小为3*3，激活函数为relu
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        # 池化层3，最大池化，对2*2的区域进行池化操作
        tf.keras.layers.MaxPooling2D(2, 2),

        # 将输出转化为一维
        tf.keras.layers.Flatten(),
        # 构建一个具有128个神经元的全连接层，激活函数使用relu
        tf.keras.layers.Dense(128, activation='relu'),   # 将前面提取的特征，在dense经过非线性变化，提取这些特征之间的关联，最后映射到输出空间上
        # 加入dropout，防止过拟合。
        tf.keras.layers.Dropout(0.4),
        # 通过softmax函数将模型输出为类名长度的神经元上，激活函数采用softmax对应概率值
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

    # 构建cnn模型
    model = get_model(len(class_names))
    # 开始训练
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    # 保存训练模型
    model.save("models/monkey_cnn.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print("程序运行时间: ", run_time, "s")

    # 绘制模型训练过程图
    draw(history, "cnn")


if __name__ == '__main__':
    train()
