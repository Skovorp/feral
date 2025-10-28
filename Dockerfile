FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg 

COPY . .

RUN pip install -r requirements.txt