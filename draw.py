import matplotlib.pyplot as plt


# 画出损失曲线
def draw(history, photo_name):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 画图
    # 设置绘图识别中文
    plt.rc("font", family='Microsoft YaHei')

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='训练集准确度')
    plt.plot(val_acc, label='验证集准确度')
    plt.legend(loc='lower right')
    plt.ylabel('准确度')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('训练集和验证集的准确度')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='训练集损失度')
    plt.plot(val_loss, label='验证集损失度')
    plt.legend(loc='upper right')
    plt.ylabel('交叉熵')
    plt.title('训练集合验证集的损失度')
    plt.xlabel('epoch')
    plt.savefig('results/' + photo_name + '.png', dpi=100)
