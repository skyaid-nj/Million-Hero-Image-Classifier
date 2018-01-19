#coding: utf-8
import tensorflow as tf
from CNN_model import model_cnn
from Data_Prepare import Data_Prepare
import os
import fire

cur_dir_path = os.path.dirname(__file__)

class model_train:

    def __init__(self):
        self.data_parepare = Data_Prepare()
        self.model_prepare = model_cnn()


    def train_main(self, log_dir_name):
        '''
        训练cnn模型
        :param log_dir_name: 本地运行保存summary的目录名
        :return:
        '''
        iter_num = 1000
        batch_size = 32
        capacity = 500
        min_after_dequeue = 499
        logs_path = os.path.join(cur_dir_path, 'logs/', log_dir_name)

        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
        y = tf.placeholder(tf.float32, shape=[None, 2])
        keep_prob = tf.placeholder(tf.float32)

        image_batch, label_batch = self.data_parepare.generate_batch(batch_size, capacity, min_after_dequeue)

        [y_out_prob, y_out_label] = self.model_prepare.forward_process(x, keep_prob)
        loss_cross_entropy = self.model_prepare.calculate_loss_cross_entropy(y_out_prob, y)
        batch_error_rate = self.model_prepare.calculate_batch_error_rate(y_out_prob, y)
        train_step = self.model_prepare.backpropagation_process(y_out_prob, y)

        # 保存
        saver = tf.train.Saver(max_to_keep=1)
        # Create a summary to monitor cost tensor
        loss_cross_entropy_summary = tf.summary.scalar("loss_cross_entropy", loss_cross_entropy)
        batch_error_rate_summary = tf.summary.scalar("error_rate", batch_error_rate)
        # Merge special summaries into a single op
        merged_summary_op = tf.summary.merge([loss_cross_entropy_summary, batch_error_rate_summary])

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # op to write logs to Tensorboard
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

            for iter in range(iter_num):
                print(iter)
                image_batch_train, label_batch_train = sess.run([image_batch, label_batch])
                [cur_iter_error, cur_batch_error_rate, summary, _] = sess.run([loss_cross_entropy, batch_error_rate, merged_summary_op, train_step], feed_dict = {x : image_batch_train, y : label_batch_train, keep_prob: 0.5})
                summary_writer.add_summary(summary, iter)
            # 模型持久化
            save_path = saver.save(sess, "./Model/model_cnn.ckpt")


if __name__ == '__main__':
    fire.Fire(model_train)