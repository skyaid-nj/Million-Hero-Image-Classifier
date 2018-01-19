# coding:utf-8
import tensorflow as tf
from CNN_model import model_cnn
import os
from Data_Prepare import Data_Prepare
import fire

class model_test:

    def __init__(self):
        model_cnn_obj = model_cnn()
        self.data_prepare_obj = Data_Prepare()
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
        self.keep_prob = tf.placeholder(tf.float32)

        [self.y_out_prob, self.y_out_label] = model_cnn_obj.forward_process(self.x, self.keep_prob)

        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, "./Model/model_cnn.ckpt")


    def read_image_from_local(self, path):
        image = tf.read_file(path)
        standardization_tensor = self.data_prepare_obj.image_preprocess(image)
        standardization_tensor = tf.reshape(standardization_tensor, shape=[-1, 28, 28, 3])
        return standardization_tensor


    def run_one_image(self, path):
        '''
        给定本地的图片文件的路径，识别
        :param path:
        :return:
        '''
        image = self.sess.run(self.read_image_from_local(path))
        [y_out_prob, y_out_label] = self.sess.run([self.y_out_prob, self.y_out_label], feed_dict = {self.x : image, self.keep_prob: 1})
        return y_out_label

    def run_on_dir(self, dir_path):
        '''
        给定一个目录，识别下面的图片文件
        :param dir_path:
        :return:
        '''
        res_list = []
        for file_name in os.listdir(dir_path):
            cur_file_path = os.path.join(dir_path, file_name)
            cur_y_out_label = self.run_one_image(cur_file_path)
            res_list.append(cur_y_out_label.tolist())
        print(res_list)

if __name__ == '__main__':
    fire.Fire(model_test)

    '''
    cur_dir_path = os.path.dirname(__file__)
    test_obj = model_test()
    print("===================== test on image path:==============================")
    path = os.path.join(cur_dir_path, 'finish_answer', 'screenshot_1516021508.82.png')
    test_obj.run_one_image(path)
    print('\n'*2)
    print("===================== test on image dir:==============================")
    path = os.path.join(cur_dir_path, 'finish_answer')
    test_obj.run_on_dir(path)
    '''