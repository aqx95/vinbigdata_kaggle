#!/bin/bash

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# install mmcv-full thus we could use CUDA operators
pip install mmcv-full

# Install mmdetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

# install Pillow 7.0.0 back in order to avoid bug in colab
#!pip install Pillow==7.0.0
pip install -r requirements/build.txt
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install -v -e .  # or "python setup.py develop"
