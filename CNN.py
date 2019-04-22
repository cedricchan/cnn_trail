# coding=utf-8
import pickle  # 用于序列化和反序列化
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math

'''
字典形式的数据：
cifar100 data content: 
    { 
    "data" : [(R,G,B, R,G,B ,....),(R,G,B, R,G,B, ...),...]    # 50000张图片，每张： 32 * 32 * 3
    "coarse_labels":[0,...,19],                         # 0~19 super category 
    "filenames":["volcano_s_000012.png",...],   # 文件名
    "batch_label":"", 
    "fine_labels":[0,1...99]          # 0~99 category 
    } 
'''


class Cifar100DataReader():
    def __init__(self, cifar_folder, onehot=True):
        self.cifar_folder = cifar_folder
        self.onehot = onehot
        self.data_label_train = None  # 训练集
        self.data_label_test = None  # 测试集
        self.batch_index = 0  # 训练数据的batch块索引
        self.test_batch_index = 0  # 测试数据的batch_size
        f = os.path.join(self.cifar_folder, "train")  # 训练集有50000张图片，100个类，每个类500张
        print('read: %s' % f)
        fo = open(f, 'rb')
        self.dic_train = pickle.load(fo, encoding='bytes')
        fo.close()
        self.data_label_train = list(zip(self.dic_train[b'data'], self.dic_train[b'fine_labels']))  # label 0~99
        np.random.shuffle(self.data_label_train)

    def dataInfo(self):
        print(self.data_label_train[0:2])  # 每个元素为二元组，第一个是numpy数组大小为32*32*3，第二是label
        print(self.dic_train.keys())
        print(b"coarse_labels:", len(self.dic_train[b"coarse_labels"]))
        print(b"filenames:", len(self.dic_train[b"filenames"]))
        print(b"batch_label:", len(self.dic_train[b"batch_label"]))
        print(b"fine_labels:", len(self.dic_train[b"fine_labels"]))
        print(b"data_shape:", np.shape((self.dic_train[b"data"])))
        print(b"data0:", type(self.dic_train[b"data"][0]))

    # 得到下一个batch训练集，块大小为100
    def next_train_data(self, batch_size=100):
        """ 
        return list of numpy arrays [na,...,na] with specific batch_size 
                na: N dimensional numpy array  
        """
        if self.batch_index < len(self.data_label_train) / batch_size:
            print("batch_index:", self.batch_index)
            datum = self.data_label_train[self.batch_index * batch_size:(self.batch_index + 1) * batch_size]
            self.batch_index += 1
            return self._decode(datum, self.onehot)
        else:
            self.batch_index = 0
            np.random.shuffle(self.data_label_train)
            datum = self.data_label_train[self.batch_index * batch_size:(self.batch_index + 1) * batch_size]
            self.batch_index += 1
            return self._decode(datum, self.onehot)

            # 把一个batch的训练数据转换为可以放入神经网络训练的数据

    def _decode(self, datum, onehot):
        rdata = list()  # batch训练数据
        rlabel = list()
        if onehot:
            for d, l in datum:
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))  # 转变形状为：32*32*3
                hot = np.zeros(100)
                hot[int(l)] = 1  # label设为100维的one-hot向量
                rlabel.append(hot)
        else:
            for d, l in datum:
                rdata.append(np.reshape(np.reshape(d, [3, 1024]).T, [32, 32, 3]))
                rlabel.append(int(l))
        return rdata, rlabel

        # 得到下一个测试数据 ，供神经网络计算模型误差用

    def next_test_data(self, batch_size=100):
        ''''' 
        return list of numpy arrays [na,...,na] with specific batch_size 
                na: N dimensional numpy array  
        '''
        if self.data_label_test is None:
            f = os.path.join(self.cifar_folder, "test")
            print('read: %s' % f)
            fo = open(f, 'rb')
            dic_test = pickle.load(fo, encoding='bytes')
            fo.close()
            data = dic_test[b'data']
            labels = dic_test[b'fine_labels']  # 0 ~ 99
            self.data_label_test = list(zip(data, labels))
            self.batch_index = 0

        if self.test_batch_index < len(self.data_label_test) / batch_size:
            print("test_batch_index:", self.test_batch_index)
            datum = self.data_label_test[self.test_batch_index * batch_size:(self.test_batch_index + 1) * batch_size]
            self.test_batch_index += 1
            return self._decode(datum, self.onehot)
        else:
            self.test_batch_index = 0
            np.random.shuffle(self.data_label_test)
            datum = self.data_label_test[self.test_batch_index * batch_size:(self.test_batch_index + 1) * batch_size]
            self.test_batch_index += 1
            return self._decode(datum, self.onehot)

            # 显示 9张图像

    def showImage(self):
        rdata, rlabel = self.next_train_data()
        fig = plt.figure()
        ax = fig.add_subplot(331)
        ax.imshow(rdata[0])
        ax = fig.add_subplot(332)
        ax.imshow(rdata[1])
        ax = fig.add_subplot(333)
        ax.imshow(rdata[2])
        ax = fig.add_subplot(334)
        ax.imshow(rdata[3])
        ax = fig.add_subplot(335)
        ax.imshow(rdata[4])
        ax = fig.add_subplot(336)
        ax.imshow(rdata[5])
        ax = fig.add_subplot(337)
        ax.imshow(rdata[6])
        ax = fig.add_subplot(338)
        ax.imshow(rdata[7])
        ax = fig.add_subplot(339)
        ax.imshow(rdata[8])
        plt.show()


# 定义卷积神经网络模型
def CNN():
    sess = tf.InteractiveSession()

    max_steps = 3000  # 最大迭代次数
    batch_size = 50  # 每次迭代的样本数量

    # 设置CNN的输入值
    image_holder = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])  # 图像大小：32 * 32 * 3
    label_holder = tf.placeholder(tf.float32, [batch_size, 100])

    # 创建第一个卷积层
    weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)  # 64个5*5*3的卷积核初始化
    kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')  # 卷积操作,步长为1
    bias1 = tf.Variable(tf.constant(0.0, shape=[64]))  # 偏置初始化
    conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))  # 加上偏置，代入激活函数
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 池化操作，步长为2
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  # LRN层

    # 创建第二个卷积层
    weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)  # 64个5*5*64的卷积核初始化
    kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')  # 卷积操作，步长为1
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))  # 初始化偏置
    conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))  # 加上偏置，代入激活函数
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)  # LRN层
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # 池化操作步长为2

    # 全连接层1
    reshape = tf.reshape(pool2, [batch_size, -1])  # 对上一层结果展开为一维
    dim = reshape.get_shape()[1].value  # 获取一维向量的大小
    weight3 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)  # 下一层有384个单元
    bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
    local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

    # 全连接层2
    weight4 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)  # 下一层有192个单元
    bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
    local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

    # 最后一层softmax层
    weight5 = variable_with_weight_loss(shape=[192, 100], stddev=1 / 192.0, w1=0.0)  # 输入100维的向量
    bias5 = tf.Variable(tf.constant(0.0, shape=[100]))
    logits = tf.nn.softmax(tf.matmul(local4, weight5) + bias5)

    # 定义损失函数和优化器       
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_holder * tf.log(logits), reduction_indices=[1]))
    tf.add_to_collection('losses', cross_entropy)  # 将交叉熵加入损失函数集合losses
    loss = tf.add_n(tf.get_collection('losses'))  # 将losses全部结果相加

    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 定义优化器
    # top_k_op=tf.nn.in_top_k(logits,label_holder,1)  # 返回一个向量，向量长度为样本点个数
    '''
    函数原型：in_top_k(predictions, targets, k, name=None)
      predictions：预测的结果，预测矩阵大小为样本数×标注的label类的个数的二维矩阵。
      targets：实际的标签，大小为样本数。
      k：每个样本的预测结果的前k个最大的数里面是否包含targets预测中的标签，一般都是取1，
         即取预测最大概率的索引与标签对比。
      top_1_op(k=1)为True的地方top_2_op(k=2)一定为True，top_1_op取样本的最大预测概率的索引与实际标签对比，
    top_2_op取样本的最大和仅次最大的两个预测概率与实际标签对比，如果实际标签在其中则为True，否则为False。
    其他k的取值可以类推。
    '''

    # 开始训练模型
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print("Training begin......")
    cifar100 = Cifar100DataReader(cifar_folder="E:/testdata/cifar-100")
    for step in range(max_steps):
        start = time.time()
        image_batch, label_batch = cifar100.next_train_data(batch_size=batch_size)
        train_op.run(feed_dict={image_holder: image_batch, label_holder: label_batch})

    print("training end.")
    print("caculate precision......")

    # 计算测试集上的误差率
    num_example = 10000  # 测试集有1000张图片
    num_iter = int(math.ceil(num_example / batch_size))  # 最大迭代次数
    true_count = 0
    total_sample_count = num_iter * batch_size
    step = 0
    while step < num_iter:
        test_data, test_label = cifar100.next_test_data(batch_size=batch_size)
        correction_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label_holder, 1))
        correction = sess.run([correction_prediction], feed_dict={image_holder: test_data, label_holder: test_label})
        true_count += np.sum(correction)
        step += 1
    precision = true_count / total_sample_count
    print("precision:", precision)

    # 保存模型
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./Cifar100/model.ckpt")
    print("save model:{0} Finished".format(save_path))


# 定义损失函数(可以作为上面损失函数的替换)
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_colletion('losses'), name='tatol_loss')


# 定义初始化weight的函数，计算weight的L2范数，并作为损失函数中的正则化项
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')  # w1表示正则化项的权重
        tf.add_to_collection('losses', weight_loss)  # 把正则化项统一存到collection中，最后加入目标函数
    return var


if __name__ == '__main__':
    CNN()
