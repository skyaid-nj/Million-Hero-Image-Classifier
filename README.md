# Million-Hero-Image-Classifier

## image build:
docker build -t tensorflow-py3-flask-env .

## container:
docker run -it -v ~/shadow/image_classifier:/opt/image_classifier -p 5400:5400 -p 5401:6006 tensorflow-py3-flask-env bash

## run:
* python3 CNN_train.py
* python3 app.py

## visualization
tensorboard --logdir='./logs' --host=0.0.0.1
打开浏览器输入：http:127.0.0.1/5401