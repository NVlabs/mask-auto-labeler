FROM nvcr.io/nvidian/pytorch:21.12-py3
RUN apt-get update
RUN apt-get install -y htop vim tmux gcc g++ psmisc iputils-ping
RUN apt-get install -y libgl1 zip
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 --force
RUN pip install opencv-python==4.5.5.64
RUN pip install scipy pytorch_lightning torchmetrics mmcv-full mmdet mmcls
RUN pip install einops shapely
RUN pip install git+https://github.com/lvis-dataset/lvis-api.git
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
