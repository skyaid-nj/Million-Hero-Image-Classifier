FROM ubuntu:16.04

RUN \
  export TZ="Asia/Shanghai" && \
  sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
  sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
  apt-get update -y &&\
  apt-get install \
    python3 \
    curl \
    python3-pip -y \
    vim \
    python3-tk
RUN \
  ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
  curl https://phuslu.github.io/bashrc >/root/.bashrc

RUN \
    cp ./requirements.txt /opt/image_classifier/
    pip install -r requirements.txt


WORKDIR /opt/image_classifier/
