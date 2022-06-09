import os
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 指定数据位置
base_path = "./data/monkey"
train_path = os.path.join(base_path, "train")
validation_path = os.path.join(base_path, "validation")

# 设置图片的宽高
image_height = 300
image_width = 300
# 一次训练所选取的样本数，迭代次数
batch_size = 256
epochs = 10


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

    return total_train, total_val
