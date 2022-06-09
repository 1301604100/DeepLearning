import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import *

# 指定数据位置
base_path = "./data/monkey"
train_path = base_path + "/train"
validation_path = base_path + "/validation"

# 设置图片的宽高
image_height = 300
image_width = 300
# 一次训练所选取的样本数，迭代次数
batch_size = 256
epochs = 10
# 分类
class_num = 10


def load_data():
    # 定义训练集图像生成器，并进行图像增强
    train_image_generator = ImageDataGenerator(rescale=1. / 255,  # 归一化
                                               rotation_range=40,  # 旋转范围
                                               width_shift_range=0.2,  # 水平平移范围
                                               height_shift_range=0.2,  # 垂直平移范围
                                               shear_range=0.2,  # 剪切变换的程度
                                               zoom_range=0.2,  # 剪切变换的程度
                                               horizontal_flip=True,  # 水平翻转
                                               fill_mode='nearest')  # 最近邻算法填补移动后图像的空白

    # 使用图像生成器读取样本，对标签进行one-hot编码
    train_data_gen = train_image_generator.flow_from_directory(directory=train_path,  # 从训练集路径读取图片
                                                               batch_size=batch_size,  # 一次训练所选取的样本数
                                                               shuffle=True,  # 打乱标签
                                                               target_size=(image_height, image_width),  # 图片resize大小
                                                               class_mode='categorical')  # one-hot编码

    # 训练集样本数
    total_train = train_data_gen.n

    # 定义验证集图像生成器，并对图像进行预处理
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # 归一化

    # 使用图像生成器从验证集validation_dir中读取样本
    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_path,  # 从验证集路径读取图片
                                                                  batch_size=batch_size,  # 一次训练所选取的样本数
                                                                  shuffle=False,  # 不打乱标签
                                                                  target_size=(image_height, image_width),
                                                                  # 图片resize大小
                                                                  class_mode='categorical')  # one-hot编码

    # 验证集样本数
    total_val = val_data_gen.n
    # print(train_data_gen.classes[1:10])
    return train_data_gen, val_data_gen,  total_train, total_val


def get_model():
    # 搭建模型
    model = tf.keras.models.Sequential([
        # 对模型做归一化的处理，将0-255之间的数字统一处理到0到1之间
        # tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        # 卷积层，该卷积层的输出为32个通道，卷积核的大小是3*3，激活函数为relu
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # 添加池化层，池化的kernel大小是2*2
        tf.keras.layers.MaxPooling2D(2, 2),
        # Add another convolution
        # 卷积层，输出为64个通道，卷积核大小为3*3，激活函数为relu
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # 池化层，最大池化，对2*2的区域进行池化操作
        tf.keras.layers.MaxPooling2D(2, 2),
        # 将二维的输出转化为一维
        tf.keras.layers.Flatten(),
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(128, activation='relu'),
        # 通过softmax函数将模型输出为类名长度的神经元上，激活函数采用softmax对应概率值
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # 输出模型信息
    model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数
    model.compile(optimizer='sgd', loss='spares_categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model





if __name__ == '__main__':
    load_data()
