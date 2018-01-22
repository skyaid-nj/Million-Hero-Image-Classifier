# Million-Hero-Image-Classifier

## 若只想调用api，已开放有http api, 目前只支持png格式
参考的python代码如下：
```
import requests

image_path = '/home/ubuntu/tensorflow_flask_server/test_data/positive/screenshot_1516018333.56.png'
files = {'raw_image': open(image_path, 'rb')}
r = requests.post("http://111.231.132.34:5000/image_classifier_one_image", files=files)

print(r.text)
```

## clone repo
git clone https://adc.github.trendmicro.com/Consumer-SkyAid/Million-Hero-Image-Classifier.git <br>
cd Million-Hero-Image-Classifier/

## image build:
docker build -t tensorflow-py3-flask-env .

## container run:
docker run -it -p 5000:5000 -p 6006:6006 tensorflow-py3-flask-env bash

## procedure run:
### model:

#### train:
* python3 CNN_train.py train_main --log_dir_name=[log_dir] <br>
eg: python3 CNN_train.py train_main --log_dir_name=train

#### test
* python3 CNN_test.py run_one_image --image_path=[image_path] <br>
eg: python3 CNN_test.py run_one_image --path=./test_data/finish_answer/screenshot_1516021508.82.png

* python3 CNN_test.py run_on_dir --dir_path=[image_dir] <br>
eg: python3 CNN_test.py run_on_dir --dir_path=./test_data/finish_answer/

### server:
* python3 app.py main --host=127.0.0.1 --port=5000 <br>
server运行成功，就可以以http协议，form-data的形式发送图片给server。server返回识别信息。

#### terminal command:
* cd [image-dir]
* curl -F "raw_image=@[image-file-path]" http://localhost:5000/image_classifier_one_image

## tensorboard visualization
tensorboard --logdir='./logs/[log_dir]' --host=127.0.0.1 <br>
* note:这里的[log_dir]就是训练过程设置的--log_dir_name=[log_dir]
* eg: tensorboard --logdir=./logs/train --host=127.0.0.1 <br>
打开浏览器输入：http:127.0.0.1:6006
