# coding: utf-8
import tensorflow as tf
import os


class Data_Prepare:
    '''
    类Data_Prepare用于数据准备工作
    '''

    def __init__(self):
        pass

    def filename_label_pipeline(self):
        '''
        生成文件名和标签列表
        :param file_path: 数据存储路径
        :return: file list 和 label list
        '''

        cur_dir_path = os.getcwd()
        data_dir = ''
        if os.path.exists(os.path.join(cur_dir_path, 'Image')):
            data_dir = os.path.join(cur_dir_path, 'Image')
        else:
            print("Cannot find data dir !!!")

        label_list = []
        filename_list = []
        label_iter = 0
        for dir in os.listdir(data_dir):
            print('dir info: {}'.format(dir))
            label_vec = [0] * len(os.listdir(data_dir))
            label_vec[label_iter] = 1
            for image_file in os.listdir(os.path.join(data_dir,dir)):
                label_list.append(label_vec)
                filename_list.append(os.path.join(data_dir, dir, image_file))
            label_iter += 1

        return label_list, filename_list

        pass

    def generate_batch(self, batch_size, capacity, min_after_dequeue):
        '''
        生成用于训练的batch化的数据
        :param label_list:
        :param file_list:
        :return:
        '''

        labels, images = self.filename_label_pipeline()
        images = tf.cast(images, tf.string)
        labels = tf.cast(labels, tf.int32)
        image_label_queue = tf.train.slice_input_producer([images, labels])
        image_queue = tf.read_file(image_label_queue[0])
        standardization_tensor_queue = self.image_preprocess(image_queue)
        image_batch, label_batch = tf.train.shuffle_batch([standardization_tensor_queue, image_label_queue[1]], \
                                            batch_size=batch_size, capacity=capacity, \
                                            min_after_dequeue=min_after_dequeue)

        label_batch = tf.reshape(label_batch, [batch_size, 2])
        image_batch = tf.cast(image_batch, tf.float32)
        return image_batch, label_batch


    def image_preprocess(self, image_queue):
        '''
        此函数用于图片的一些预处理，主要用于裁剪，resize，标准化
        :param image:
        :return:
        '''
        with tf.name_scope('image_preprocess'):
            pixel_tensor_queue = tf.image.decode_png(image_queue, channels=3)
            img_shape_queue = tf.shape(pixel_tensor_queue)
            img_height_queue = tf.to_int32(tf.to_float(img_shape_queue[0]) * 0.6)
            img_width_queue = img_shape_queue[1]
            cropped_pixel_tensor_queue = tf.image.crop_to_bounding_box(pixel_tensor_queue, 20, 20, img_height_queue, img_width_queue - 20)
            resized_cropped_pixel_tensor_queue = tf.image.resize_images(cropped_pixel_tensor_queue, (28, 28))
            standardization_tensor_queue = tf.image.per_image_standardization(resized_cropped_pixel_tensor_queue)
        return standardization_tensor_queue


if __name__ == '__main__':
    data_prepare = Data_Prepare()
    label_list, filename_list = data_prepare.filename_label_pipeline()
    print('label_list is {}'.format(label_list))
    print('filename_list is {}'.format(filename_list))

