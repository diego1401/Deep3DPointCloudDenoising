ARG PYTORCH="1.9.1"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install tensorboard
RUN pip3 install numpy
RUN pip3 install plyfile
RUN pip3 install sklearn
RUN pip3 install trimesh
RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
RUN pip3 install easydict
RUN pip3 install scipy
RUN pip3 install rtree
RUN pip3 uninstall -y protobuf
RUN pip3 install protobuf==3.20.0
# RUN pip3 install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu111_pyt1100/download.html
#ADD . .
RUN pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
#RUN chmod +x compile_ops.sh 
