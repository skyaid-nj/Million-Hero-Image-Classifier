#coding: utf-8
import tensorflow as tf

class model_cnn:
    def __init__(self):
        pass

    def forward_process(self, x, keep_prob):
        '''
        cnn 模型的前向过程
        :param x: 输入数据
        :return:
        '''

        with tf.name_scope('conv1'):
            conv1_w = tf.get_variable('conv1_w', shape = [5, 5, 3, 32], dtype = tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            conv1_b = tf.get_variable('conv1_b', shape=[32], dtype = tf.float32, initializer=tf.constant_initializer(0.1))
            h1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
            h1_relu = tf.nn.relu(h1)

        with tf.name_scope('max_pool1'):
            h1_pool = tf.nn.max_pool(h1_relu, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], \
                                      padding='SAME')

        with tf.name_scope('conv2'):
            conv2_w = tf.get_variable('conv2_w', shape = [3, 3, 32, 64], dtype = tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            conv2_b = tf.get_variable('conv2_b', shape=[64], dtype = tf.float32, initializer=tf.constant_initializer(0.1))
            h2 = tf.nn.conv2d(h1_pool, conv2_w, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
            h2_relu = tf.nn.relu(h2)

        with tf.name_scope('max_pool2'):
            h2_pool = tf.nn.max_pool(h2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], \
                                      padding='SAME')

        with tf.name_scope('fc1'):
            fc1_w = tf.get_variable('fc1_w', shape = [7*7*64, 128], dtype = tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            fc1_b = tf.get_variable('fc1_b', shape = [128], dtype = tf.float32, initializer=tf.constant_initializer(0.1))
            h2_pool_flat = tf.reshape(h2_pool, [-1, 7*7*64])
            h3_relu = tf.nn.relu(tf.matmul(h2_pool_flat, fc1_w) + fc1_b)
            h3_relu_drop = tf.nn.dropout(h3_relu, keep_prob)

        with tf.name_scope('f2'):
            fc2_w = tf.get_variable('fc2_w', shape = [128, 2], dtype = tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
            fc2_b = tf.get_variable('fc2_b', shape = [2], dtype = tf.float32, initializer=tf.constant_initializer(0.1))
            h4 = tf.matmul(h3_relu_drop, fc2_w) + fc2_b
            h4_softmax = tf.nn.softmax(h4)

        y_out_prob = h4_softmax
        y_out_label = tf.argmax(y_out_prob, axis = 1)
        return y_out_prob, y_out_label


    def calculate_loss_cross_entropy(self, y_out, y):
        '''
        计算样本集的预测错误率,基于交叉熵
        :param y_out: 模型输出
        :param y: 实际标签
        :return:
        '''
        with tf.name_scope('calculate_loss_cross_entropy'):
            return -tf.reduce_mean(y * tf.log(y_out))

    def calculate_batch_error_rate(self, y_out, y):
        '''
        计算batch的预测错误率,基于真实输出和预测值的差异
        :param y_out: 模型输出
        :param y: 实际标签
        :return:
        '''
        with tf.name_scope('calculate_batch_error_rate'):
            acc = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            return 1 - acc

    def backpropagation_process(self, y_out, y):
        '''
        反馈更新权值过程
        :param y_out: forward_process的输出
        :param y: 真实值
        :return:
        '''

        with tf.name_scope('backpropagation_process'):
            cross_entropy = self.calculate_loss_cross_entropy(y_out, y)
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        return train_step

