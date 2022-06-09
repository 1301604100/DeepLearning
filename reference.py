# ����1��
# ѵ����������������Ѿ��������ã���ѵ������Ŀ¼�� c:\training, ���Լ���Ŀ¼��c:\test
# ѵ������Ŀ¼��ϸ�ּ����ࣺ �磬 c:\training\cat      c:\training\dog      c:\training\mouse...
# ���Լ�Ҳһ������ʾ��Ӧ����Ŀ¼�洢 ��Ӧ�����ͼƬ
#
# ��������·���������ȡѵ��������Լ����ݣ�

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = r'c:\training'
validation_path = r'c:\test'

image_generator = ImageDataGenerator(rescale=1. / 255.)

im_height = 224   # ��Ҫ��ȫ��ͼƬͳһ�� �����ĳߴ磬��ʵ��Ҫ������
im_width = 244

# ѵ����������������one-hot���룬��������
train_data_gen = image_generator.flow_from_directory(directory=train_path,
                                                     # batch_size=batch_size,
                                                     shuffle=False,
                                                     target_size=(im_height, im_width),
                                                     class_mode='categorical')
# �������0,1,2,3, ... �� c:\training ��Ŀ¼���Ƴ��ֵ��ֵ�˳��                                         
                                                    
# ���Լ�������������one-hot����
valid_data_gen = image_generator.flow_from_directory(directory=validation_path,
                                                     # batch_size=batch_size *10,
                                                     shuffle=False,
                                                     target_size=(im_height, im_width),
                                                     class_mode='categorical')
                                                     
                                                     
# �õ�֮��
# model = Sequential([ ... ])

# ѵ��ʱ��
# steps_per_epoch = train_data_gen.n // train_data_gen.batch_size      #����ÿ��epochҪ�����ͼƬ����
# history = model.fit(train_data_gen,
#                    steps_per_epoch=steps_per_epoch,
#                     # epochs=training_epoches,
#                     verbose=1, shuffle=True)
                    
# ���۲��Լ�
steps = valid_data_gen.n // valid_data_gen.batch_size
# model.evaluate(valid_data_gen, steps=steps)

# ���߲���
# model.predict(valid_data_gen)


#
# ����2��
# ѵ��������Լ�������û�зֿ����綼��һ��Ŀ¼����c:\data
# ��������Ŀ¼����c:\data\cat, c:\data\dog, c:\data\mouse...
 
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator



validation_split = 0.3    # ��ʾ�������30% �����Լ�

image_generator = ImageDataGenerator(rescale=1. / 255., validation_split=validation_split)

#Ϊ��ʹÿ�ε� ѵ��������Լ��̶�����Ҫ���ù̶����������
np.random.seed(1000)  # �������

im_height = 224   # ��Ҫ��ȫ��ͼƬͳһ�� �����ĳߴ磬��ʵ��Ҫ������
im_width = 244

# ѵ����������������one-hot���룬ע�⣬�ȷ���1����һ������subset
# train_data_gen = image_generator.flow_from_directory(directory=image_path,
#                                                      batch_size=batch_size,
#                                                      shuffle=False,
#                                                      target_size=(im_height, im_width),
#                                                      class_mode='categorical',
#                                                      subset='training')
# ���Լ�������������one-hot����
# valid_data_gen = image_generator.flow_from_directory(directory=image_path,
#                                                      batch_size=batch_size,
#                                                      shuffle=False,
#                                                      target_size=(im_height, im_width),
#                                                      class_mode='categorical',
#                                                      subset='validation')
                                   
                                   
# ģ�͵�ѵ������������뷽��1һ��
#
# �´��ٸ�������3�뷽��4