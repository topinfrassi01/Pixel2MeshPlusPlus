FROM tensorflow/tensorflow:1.13.2-gpu-py3
COPY Pixel2MeshPlusPlus/requirements.txt .
WORKDIR /
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
