FROM nvcr.io/nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python3.8 python3-dev python3-pip
RUN python3.8 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt


CMD [ "/bin/bash" ]
