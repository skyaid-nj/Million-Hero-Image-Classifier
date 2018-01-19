# Million-Hero-Image-Classifier

## image build:
docker build -t tensorflow-py3-flask-env .

## container run:
docker run -it -v ~/shadow/image_classifier:/opt/image_classifier -p 5000:5000 -p 6006:6006 tensorflow-py3-flask-env bash

## procedure run:
### model:

#### train:
* python3 CNN_train.py train_main --log_dir_name=[name(任意字符串)]

#### test
* python3 CNN_test.py run_one_image --path=[image_path]
* python3 CNN_test.py run_on_dir --dir_path=[image_dir]

### server:
* python3 app.py --host=127.0.0.0 --port=5000
server运行成功，就可以以http协议，form-data的形式发送图片给server。server返回识别信息。
terminal command:
* cd [image-dir]
* curl -F "raw_image=@[image-file-path]" http://localhost:5000/image_classifier_one_image

## tensorboard visualization
tensorboard --logdir='./logs' --host=127.0.0.0
打开浏览器输入：http:127.0.0.1:6006