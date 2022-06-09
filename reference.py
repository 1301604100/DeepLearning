# 方法1：
# 训练数据与测试数据已经单独放置，如训练集的目录在 c:\training, 测试集的目录在c:\test
# 训练集的目录有细分几个类： 如， c:\training\cat      c:\training\dog      c:\training\mouse...
# 测试集也一样，表示相应的子目录存储 相应的类别图片
#
# 则采用以下方法单独读取训练集与测试集数据：

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = r'c:\training'
validation_path = r'c:\test'

image_generator = ImageDataGenerator(rescale=1. / 255.)

im_height = 224   # 需要把全部图片统一的 长与宽的尺寸，按实际要求设置
im_width = 244

# 训练集数据生成器，one-hot编码，打乱数据
train_data_gen = image_generator.flow_from_directory(directory=train_path,
                                                     # batch_size=batch_size,
                                                     shuffle=False,
                                                     target_size=(im_height, im_width),
                                                     class_mode='categorical')
# 其中类别0,1,2,3, ... 按 c:\training 子目录名称出现的字典顺序                                         
                                                    
# 测试集数据生成器，one-hot编码
valid_data_gen = image_generator.flow_from_directory(directory=validation_path,
                                                     # batch_size=batch_size *10,
                                                     shuffle=False,
                                                     target_size=(im_height, im_width),
                                                     class_mode='categorical')
                                                     
                                                     
# 得到之后
# model = Sequential([ ... ])

# 训练时：
# steps_per_epoch = train_data_gen.n // train_data_gen.batch_size      #计算每个epoch要计算的图片个数
# history = model.fit(train_data_gen,
#                    steps_per_epoch=steps_per_epoch,
#                     # epochs=training_epoches,
#                     verbose=1, shuffle=True)
                    
# 评价测试集
steps = valid_data_gen.n // valid_data_gen.batch_size
# model.evaluate(valid_data_gen, steps=steps)

# 或者采用
# model.predict(valid_data_gen)


#
# 方法2：
# 训练集与测试集的数据没有分开，如都在一个目录，如c:\data
# 下面有子目录，如c:\data\cat, c:\data\dog, c:\data\mouse...
 
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator



validation_split = 0.3    # 表示拆分数据30% 做测试集

image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split)

#为了使每次的 训练集与测试集固定，需要设置固定的随机种子
np.random.seed(1000)  # 随机种子

im_height = 224   # 需要把全部图片统一的 长与宽的尺寸，按实际要求设置
im_width = 244

# 训练集数据生成器，one-hot编码，注意，比方法1多了一个参数subset
# train_data_gen = image_generator.flow_from_directory(directory=image_path,
#                                                      batch_size=batch_size,
#                                                      shuffle=False,
#                                                      target_size=(im_height, im_width),
#                                                      class_mode='categorical',
#                                                      subset='training')
# 测试集数据生成器，one-hot编码
# valid_data_gen = image_generator.flow_from_directory(directory=image_path,
#                                                      batch_size=batch_size,
#                                                      shuffle=False,
#                                                      target_size=(im_height, im_width),
#                                                      class_mode='categorical',
#                                                      subset='validation')
                                   
                                   
# 模型的训练与测试评价与方法1一致
#
# 下次再给出方法3与方法4