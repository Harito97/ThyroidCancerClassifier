# Sử dụng hình ảnh cơ sở Python 3.10
FROM python:3.10

WORKDIR .

# Tạo thư mục làm việc và clone dự án, sau đó cài đặt các gói cần thiết
RUN mkdir app && \
    git clone https://github.com/Harito97/ThyroidCancerClassifier.git app/ThyroidCancerClassifier


RUN pip install --upgrade pip
RUN pip install -r app/ThyroidCancerClassifier/requirements.txt

# Mở cổng 5000 cho ứng dụng Flask
EXPOSE 5000

# Chạy ứng dụng Flask khi khởi động container
CMD ["python", "app/ThyroidCancerClassifier/src/gui/app.py"]

# docker build -t thyroidcancerclassifier .
# để xây dựng image docker
# docker run -p 5000:5000 thyroidcancerclassifier
# để chạy container từ image vừa xây dựng với cổng 5000 của máy host được ánh xạ với cổng 5000 của container

# $ docker build -t thyroidcancerclassifier .
# sh-5.2$ docker tag thyroidcancerclassifier:latest harito97/thyroidcancerclassifier:latest 
# permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.46/images/thyroidcancerclassifier:latest/tag?repo=harito97%2Fthyroidcancerclassifier&tag=latest": dial unix /var/run/docker.sock: connect: permission denied
# sh-5.2$ sudo docker tag thyroidcancerclassifier:latest harito97/thyroidcancerclassifier:latest 
# Error response from daemon: No such image: thyroidcancerclassifier:latest
# sh-5.2$ docker build -t thyroidcancerclassifier .
# DEPRECATED: The legacy builder is deprecated and will be removed in a future release.
#             Install the buildx component to build images with BuildKit:
#             https://docs.docker.com/go/buildx/

# permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Post "http://%2Fvar%2Frun%2Fdocker.sock/v1.46/build?buildargs=%7B%7D&cachefrom=%5B%5D&cgroupparent=&cpuperiod=0&cpuquota=0&cpusetcpus=&cpusetmems=&cpushares=0&dockerfile=Dockerfile&labels=%7B%7D&memory=0&memswap=0&networkmode=default&rm=1&shmsize=0&t=thyroidcancerclassifier&target=&ulimits=%5B%5D&version=1": dial unix /var/run/docker.sock: connect: permission denied
# sh-5.2$ sudo docker build -t thyroidcancerclassifier .
# DEPRECATED: The legacy builder is deprecated and will be removed in a future release.
#             Install the buildx component to build images with BuildKit:
#             https://docs.docker.com/go/buildx/

# Sending build context to Docker daemon  7.413MB
# Step 1/7 : FROM python:3.10
# 3.10: Pulling from library/python
# e9aef93137af: Pull complete 
# 58b365fa3e8d: Pull complete 
# 3dbed71fc544: Pull complete 
# ae70830af8b6: Pull complete 
# fd155188cf05: Pull complete 
# f0d61d0facb6: Pull complete 
# 726c326e9564: Pull complete 
# 331871c631e8: Pull complete 
# Digest: sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2
# Status: Downloaded newer image for python:3.10
#  ---> 6905cae1f812
# Step 2/7 : WORKDIR .
#  ---> Running in 626cddea81a4
#  ---> Removed intermediate container 626cddea81a4
#  ---> 29270da380e1
# Step 3/7 : RUN mkdir app &&     git clone https://github.com/Harito97/ThyroidCancerClassifier.git app/ThyroidCancerClassifier
#  ---> Running in ae28c7f3b73b
# Cloning into 'app/ThyroidCancerClassifier'...
#  ---> Removed intermediate container ae28c7f3b73b
#  ---> 557b892babb7
# Step 4/7 : RUN pip install --upgrade pip
#  ---> Running in 7246b8d55022
# Requirement already satisfied: pip in /usr/local/lib/python3.10/site-packages (23.0.1)
# Collecting pip
#   Downloading pip-24.1.2-py3-none-any.whl (1.8 MB)
#      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 12.5 MB/s eta 0:00:00
# Installing collected packages: pip
#   Attempting uninstall: pip
#     Found existing installation: pip 23.0.1
#     Uninstalling pip-23.0.1:
#       Successfully uninstalled pip-23.0.1
# WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
# Successfully installed pip-24.1.2
#  ---> Removed intermediate container 7246b8d55022
#  ---> 683285bb0f0c
# Step 5/7 : RUN pip install -r app/ThyroidCancerClassifier/requirements.txt
#  ---> Running in 1d4d6bd4812c
# Collecting Flask==3.0.3 (from -r app/ThyroidCancerClassifier/requirements.txt (line 7))
#   Downloading flask-3.0.3-py3-none-any.whl.metadata (3.2 kB)
# Collecting matplotlib==3.9.0 (from -r app/ThyroidCancerClassifier/requirements.txt (line 8))
#   Downloading matplotlib-3.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
# Collecting matplotlib-inline==0.1.7 (from -r app/ThyroidCancerClassifier/requirements.txt (line 9))
#   Downloading matplotlib_inline-0.1.7-py3-none-any.whl.metadata (3.9 kB)
# Collecting numpy==1.26.4 (from -r app/ThyroidCancerClassifier/requirements.txt (line 10))
#   Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
#      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.0/61.0 kB 911.0 kB/s eta 0:00:00
# Collecting onnx==1.16.1 (from -r app/ThyroidCancerClassifier/requirements.txt (line 11))
#   Downloading onnx-1.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (16 kB)
# Collecting onnxruntime==1.18.1 (from -r app/ThyroidCancerClassifier/requirements.txt (line 12))
#   Downloading onnxruntime-1.18.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (4.3 kB)
# Collecting onnxscript==0.1.0.dev20240710 (from -r app/ThyroidCancerClassifier/requirements.txt (line 13))
#   Downloading onnxscript-0.1.0.dev20240710-py3-none-any.whl.metadata (11 kB)
# Collecting opencv-python==4.10.0.84 (from -r app/ThyroidCancerClassifier/requirements.txt (line 14))
#   Downloading opencv_python-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
# Collecting pillow==10.3.0 (from -r app/ThyroidCancerClassifier/requirements.txt (line 15))
#   Downloading pillow-10.3.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (9.2 kB)
# Collecting scikit-learn==1.5.0 (from -r app/ThyroidCancerClassifier/requirements.txt (line 16))
#   Downloading scikit_learn-1.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (11 kB)
# Collecting scipy==1.13.1 (from -r app/ThyroidCancerClassifier/requirements.txt (line 17))
#   Downloading scipy-1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
#      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.6/60.6 kB 2.6 MB/s eta 0:00:00
# Collecting seaborn==0.13.2 (from -r app/ThyroidCancerClassifier/requirements.txt (line 18))
#   Downloading seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
# Collecting tensorboard==2.16.2 (from -r app/ThyroidCancerClassifier/requirements.txt (line 19))
#   Downloading tensorboard-2.16.2-py3-none-any.whl.metadata (1.6 kB)
# Collecting tensorboard-data-server==0.7.2 (from -r app/ThyroidCancerClassifier/requirements.txt (line 20))
#   Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl.metadata (1.1 kB)
# Collecting timm==1.0.7 (from -r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading timm-1.0.7-py3-none-any.whl.metadata (47 kB)
#      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 47.5/47.5 kB 3.0 MB/s eta 0:00:00
# Collecting torch==2.3.1 (from -r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading torch-2.3.1-cp310-cp310-manylinux1_x86_64.whl.metadata (26 kB)
# Collecting torchsummary==1.5.1 (from -r app/ThyroidCancerClassifier/requirements.txt (line 23))
#   Downloading torchsummary-1.5.1-py3-none-any.whl.metadata (296 bytes)
# Collecting torchvision==0.18.1 (from -r app/ThyroidCancerClassifier/requirements.txt (line 24))
#   Downloading torchvision-0.18.1-cp310-cp310-manylinux1_x86_64.whl.metadata (6.6 kB)
# Collecting Werkzeug>=3.0.0 (from Flask==3.0.3->-r app/ThyroidCancerClassifier/requirements.txt (line 7))
#   Downloading werkzeug-3.0.3-py3-none-any.whl.metadata (3.7 kB)
# Collecting Jinja2>=3.1.2 (from Flask==3.0.3->-r app/ThyroidCancerClassifier/requirements.txt (line 7))
#   Downloading jinja2-3.1.4-py3-none-any.whl.metadata (2.6 kB)
# Collecting itsdangerous>=2.1.2 (from Flask==3.0.3->-r app/ThyroidCancerClassifier/requirements.txt (line 7))
#   Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
# Collecting click>=8.1.3 (from Flask==3.0.3->-r app/ThyroidCancerClassifier/requirements.txt (line 7))
#   Downloading click-8.1.7-py3-none-any.whl.metadata (3.0 kB)
# Collecting blinker>=1.6.2 (from Flask==3.0.3->-r app/ThyroidCancerClassifier/requirements.txt (line 7))
#   Downloading blinker-1.8.2-py3-none-any.whl.metadata (1.6 kB)
# Collecting contourpy>=1.0.1 (from matplotlib==3.9.0->-r app/ThyroidCancerClassifier/requirements.txt (line 8))
#   Downloading contourpy-1.2.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.8 kB)
# Collecting cycler>=0.10 (from matplotlib==3.9.0->-r app/ThyroidCancerClassifier/requirements.txt (line 8))
#   Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
# Collecting fonttools>=4.22.0 (from matplotlib==3.9.0->-r app/ThyroidCancerClassifier/requirements.txt (line 8))
#   Downloading fonttools-4.53.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (162 kB)
#      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 162.6/162.6 kB 2.9 MB/s eta 0:00:00
# Collecting kiwisolver>=1.3.1 (from matplotlib==3.9.0->-r app/ThyroidCancerClassifier/requirements.txt (line 8))
#   Downloading kiwisolver-1.4.5-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.4 kB)
# Collecting packaging>=20.0 (from matplotlib==3.9.0->-r app/ThyroidCancerClassifier/requirements.txt (line 8))
#   Downloading packaging-24.1-py3-none-any.whl.metadata (3.2 kB)
# Collecting pyparsing>=2.3.1 (from matplotlib==3.9.0->-r app/ThyroidCancerClassifier/requirements.txt (line 8))
#   Downloading pyparsing-3.1.2-py3-none-any.whl.metadata (5.1 kB)
# Collecting python-dateutil>=2.7 (from matplotlib==3.9.0->-r app/ThyroidCancerClassifier/requirements.txt (line 8))
#   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
# Collecting traitlets (from matplotlib-inline==0.1.7->-r app/ThyroidCancerClassifier/requirements.txt (line 9))
#   Downloading traitlets-5.14.3-py3-none-any.whl.metadata (10 kB)
# Collecting protobuf>=3.20.2 (from onnx==1.16.1->-r app/ThyroidCancerClassifier/requirements.txt (line 11))
#   Downloading protobuf-5.27.2-cp38-abi3-manylinux2014_x86_64.whl.metadata (592 bytes)
# Collecting coloredlogs (from onnxruntime==1.18.1->-r app/ThyroidCancerClassifier/requirements.txt (line 12))
#   Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
# Collecting flatbuffers (from onnxruntime==1.18.1->-r app/ThyroidCancerClassifier/requirements.txt (line 12))
#   Downloading flatbuffers-24.3.25-py2.py3-none-any.whl.metadata (850 bytes)
# Collecting sympy (from onnxruntime==1.18.1->-r app/ThyroidCancerClassifier/requirements.txt (line 12))
#   Downloading sympy-1.13.0-py3-none-any.whl.metadata (12 kB)
# Collecting typing-extensions (from onnxscript==0.1.0.dev20240710->-r app/ThyroidCancerClassifier/requirements.txt (line 13))
#   Downloading typing_extensions-4.12.2-py3-none-any.whl.metadata (3.0 kB)
# Collecting ml-dtypes (from onnxscript==0.1.0.dev20240710->-r app/ThyroidCancerClassifier/requirements.txt (line 13))
#   Downloading ml_dtypes-0.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)
# Collecting joblib>=1.2.0 (from scikit-learn==1.5.0->-r app/ThyroidCancerClassifier/requirements.txt (line 16))
#   Downloading joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
# Collecting threadpoolctl>=3.1.0 (from scikit-learn==1.5.0->-r app/ThyroidCancerClassifier/requirements.txt (line 16))
#   Downloading threadpoolctl-3.5.0-py3-none-any.whl.metadata (13 kB)
# Collecting pandas>=1.2 (from seaborn==0.13.2->-r app/ThyroidCancerClassifier/requirements.txt (line 18))
#   Downloading pandas-2.2.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (19 kB)
# Collecting absl-py>=0.4 (from tensorboard==2.16.2->-r app/ThyroidCancerClassifier/requirements.txt (line 19))
#   Downloading absl_py-2.1.0-py3-none-any.whl.metadata (2.3 kB)
# Collecting grpcio>=1.48.2 (from tensorboard==2.16.2->-r app/ThyroidCancerClassifier/requirements.txt (line 19))
#   Downloading grpcio-1.64.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.3 kB)
# Collecting markdown>=2.6.8 (from tensorboard==2.16.2->-r app/ThyroidCancerClassifier/requirements.txt (line 19))
#   Downloading Markdown-3.6-py3-none-any.whl.metadata (7.0 kB)
# Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.10/site-packages (from tensorboard==2.16.2->-r app/ThyroidCancerClassifier/requirements.txt (line 19)) (65.5.1)
# Collecting six>1.9 (from tensorboard==2.16.2->-r app/ThyroidCancerClassifier/requirements.txt (line 19))
#   Downloading six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
# Collecting pyyaml (from timm==1.0.7->-r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
# Collecting huggingface_hub (from timm==1.0.7->-r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading huggingface_hub-0.23.4-py3-none-any.whl.metadata (12 kB)
# Collecting safetensors (from timm==1.0.7->-r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading safetensors-0.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
# Collecting filelock (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading filelock-3.15.4-py3-none-any.whl.metadata (2.9 kB)
# Collecting networkx (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading networkx-3.3-py3-none-any.whl.metadata (5.1 kB)
# Collecting fsspec (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading fsspec-2024.6.1-py3-none-any.whl.metadata (11 kB)
# Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
# Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
# Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
# Collecting nvidia-cudnn-cu12==8.9.2.26 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
# Collecting nvidia-cublas-cu12==12.1.3.1 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
# Collecting nvidia-cufft-cu12==11.0.2.54 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
# Collecting nvidia-curand-cu12==10.3.2.106 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
# Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
# Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
# Collecting nvidia-nccl-cu12==2.20.5 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl.metadata (1.8 kB)
# Collecting nvidia-nvtx-cu12==12.1.105 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.7 kB)
# Collecting triton==2.3.1 (from torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading triton-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.4 kB)
# Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch==2.3.1->-r app/ThyroidCancerClassifier/requirements.txt (line 22))
#   Downloading nvidia_nvjitlink_cu12-12.5.82-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
# Collecting MarkupSafe>=2.0 (from Jinja2>=3.1.2->Flask==3.0.3->-r app/ThyroidCancerClassifier/requirements.txt (line 7))
#   Downloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
# Collecting pytz>=2020.1 (from pandas>=1.2->seaborn==0.13.2->-r app/ThyroidCancerClassifier/requirements.txt (line 18))
#   Downloading pytz-2024.1-py2.py3-none-any.whl.metadata (22 kB)
# Collecting tzdata>=2022.7 (from pandas>=1.2->seaborn==0.13.2->-r app/ThyroidCancerClassifier/requirements.txt (line 18))
#   Downloading tzdata-2024.1-py2.py3-none-any.whl.metadata (1.4 kB)
# Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime==1.18.1->-r app/ThyroidCancerClassifier/requirements.txt (line 12))
#   Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
# Collecting requests (from huggingface_hub->timm==1.0.7->-r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading requests-2.32.3-py3-none-any.whl.metadata (4.6 kB)
# Collecting tqdm>=4.42.1 (from huggingface_hub->timm==1.0.7->-r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading tqdm-4.66.4-py3-none-any.whl.metadata (57 kB)
#      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 57.6/57.6 kB 5.7 MB/s eta 0:00:00
# Collecting mpmath<1.4,>=1.1.0 (from sympy->onnxruntime==1.18.1->-r app/ThyroidCancerClassifier/requirements.txt (line 12))
#   Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
# Collecting charset-normalizer<4,>=2 (from requests->huggingface_hub->timm==1.0.7->-r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (33 kB)
# Collecting idna<4,>=2.5 (from requests->huggingface_hub->timm==1.0.7->-r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading idna-3.7-py3-none-any.whl.metadata (9.9 kB)
# Collecting urllib3<3,>=1.21.1 (from requests->huggingface_hub->timm==1.0.7->-r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading urllib3-2.2.2-py3-none-any.whl.metadata (6.4 kB)
# Collecting certifi>=2017.4.17 (from requests->huggingface_hub->timm==1.0.7->-r app/ThyroidCancerClassifier/requirements.txt (line 21))
#   Downloading certifi-2024.7.4-py3-none-any.whl.metadata (2.2 kB)
# Downloading flask-3.0.3-py3-none-any.whl (101 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.7/101.7 kB 7.5 MB/s eta 0:00:00
# Downloading matplotlib-3.9.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.3 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.3/8.3 MB 16.9 MB/s eta 0:00:00
# Downloading matplotlib_inline-0.1.7-py3-none-any.whl (9.9 kB)
# Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 35.2 MB/s eta 0:00:00
# Downloading onnx-1.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (15.9 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 15.9/15.9 MB 35.4 MB/s eta 0:00:00
# Downloading onnxruntime-1.18.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.8 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.8/6.8 MB 36.2 MB/s eta 0:00:00
# Downloading onnxscript-0.1.0.dev20240710-py3-none-any.whl (644 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 644.3/644.3 kB 26.3 MB/s eta 0:00:00
# Downloading opencv_python-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (62.5 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.5/62.5 MB 32.9 MB/s eta 0:00:00
# Downloading pillow-10.3.0-cp310-cp310-manylinux_2_28_x86_64.whl (4.5 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 35.2 MB/s eta 0:00:00
# Downloading scikit_learn-1.5.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.3 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.3/13.3 MB 36.5 MB/s eta 0:00:00
# Downloading scipy-1.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (38.6 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 38.6/38.6 MB 25.1 MB/s eta 0:00:00
# Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 kB 22.8 MB/s eta 0:00:00
# Downloading tensorboard-2.16.2-py3-none-any.whl (5.5 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 35.6 MB/s eta 0:00:00
# Downloading tensorboard_data_server-0.7.2-py3-none-manylinux_2_31_x86_64.whl (6.6 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 36.8 MB/s eta 0:00:00
# Downloading timm-1.0.7-py3-none-any.whl (2.3 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 30.7 MB/s eta 0:00:00
# Downloading torch-2.3.1-cp310-cp310-manylinux1_x86_64.whl (779.1 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 779.1/779.1 MB 2.6 MB/s eta 0:00:00
# Downloading torchsummary-1.5.1-py3-none-any.whl (2.8 kB)
# Downloading torchvision-0.18.1-cp310-cp310-manylinux1_x86_64.whl (7.0 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.0/7.0 MB 27.9 MB/s eta 0:00:00
# Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl (410.6 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 410.6/410.6 MB 16.8 MB/s eta 0:00:00
# Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (14.1 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.1/14.1 MB 36.1 MB/s eta 0:00:00
# Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (23.7 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 35.5 MB/s eta 0:00:00
# Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (823 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 823.6/823.6 kB 34.0 MB/s eta 0:00:00
# Downloading nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 731.7/731.7 MB 2.9 MB/s eta 0:00:00
# Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl (121.6 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.6/121.6 MB 13.6 MB/s eta 0:00:00
# Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl (56.5 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.5/56.5 MB 23.7 MB/s eta 0:00:00
# Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl (124.2 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.2/124.2 MB 27.6 MB/s eta 0:00:00
# Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl (196.0 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 196.0/196.0 MB 22.8 MB/s eta 0:00:00
# Downloading nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl (176.2 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 176.2/176.2 MB 19.1 MB/s eta 0:00:00
# Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl (99 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 kB 13.3 MB/s eta 0:00:00
# Downloading triton-2.3.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (168.1 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 168.1/168.1 MB 22.9 MB/s eta 0:00:00
# Downloading absl_py-2.1.0-py3-none-any.whl (133 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.7/133.7 kB 18.6 MB/s eta 0:00:00
# Downloading blinker-1.8.2-py3-none-any.whl (9.5 kB)
# Downloading click-8.1.7-py3-none-any.whl (97 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 97.9/97.9 kB 12.5 MB/s eta 0:00:00
# Downloading contourpy-1.2.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (305 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 305.2/305.2 kB 36.0 MB/s eta 0:00:00
# Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
# Downloading fonttools-4.53.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.6/4.6 MB 18.9 MB/s eta 0:00:00
# Downloading grpcio-1.64.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (5.6 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.6/5.6 MB 30.0 MB/s eta 0:00:00
# Downloading itsdangerous-2.2.0-py3-none-any.whl (16 kB)
# Downloading jinja2-3.1.4-py3-none-any.whl (133 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 133.3/133.3 kB 17.4 MB/s eta 0:00:00
# Downloading joblib-1.4.2-py3-none-any.whl (301 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 301.8/301.8 kB 22.6 MB/s eta 0:00:00
# Downloading kiwisolver-1.4.5-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.6/1.6 MB 32.8 MB/s eta 0:00:00
# Downloading Markdown-3.6-py3-none-any.whl (105 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 105.4/105.4 kB 11.9 MB/s eta 0:00:00
# Downloading packaging-24.1-py3-none-any.whl (53 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.0/54.0 kB 6.2 MB/s eta 0:00:00
# Downloading pandas-2.2.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (13.0 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.0/13.0 MB 27.8 MB/s eta 0:00:00
# Downloading protobuf-5.27.2-cp38-abi3-manylinux2014_x86_64.whl (309 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 309.3/309.3 kB 22.9 MB/s eta 0:00:00
# Downloading pyparsing-3.1.2-py3-none-any.whl (103 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 103.2/103.2 kB 11.1 MB/s eta 0:00:00
# Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.9/229.9 kB 20.8 MB/s eta 0:00:00
# Downloading six-1.16.0-py2.py3-none-any.whl (11 kB)
# Downloading threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
# Downloading typing_extensions-4.12.2-py3-none-any.whl (37 kB)
# Downloading werkzeug-3.0.3-py3-none-any.whl (227 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 227.3/227.3 kB 23.8 MB/s eta 0:00:00
# Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.0/46.0 kB 4.4 MB/s eta 0:00:00
# Downloading filelock-3.15.4-py3-none-any.whl (16 kB)
# Downloading flatbuffers-24.3.25-py2.py3-none-any.whl (26 kB)
# Downloading fsspec-2024.6.1-py3-none-any.whl (177 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 177.6/177.6 kB 19.3 MB/s eta 0:00:00
# Downloading huggingface_hub-0.23.4-py3-none-any.whl (402 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 402.6/402.6 kB 26.3 MB/s eta 0:00:00
# Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (705 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 705.5/705.5 kB 29.9 MB/s eta 0:00:00
# Downloading ml_dtypes-0.4.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.2 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 30.1 MB/s eta 0:00:00
# Downloading networkx-3.3-py3-none-any.whl (1.7 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 29.2 MB/s eta 0:00:00
# Downloading safetensors-0.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 30.3 MB/s eta 0:00:00
# Downloading sympy-1.13.0-py3-none-any.whl (6.2 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.2/6.2 MB 34.0 MB/s eta 0:00:00
# Downloading traitlets-5.14.3-py3-none-any.whl (85 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.4/85.4 kB 8.8 MB/s eta 0:00:00
# Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.8/86.8 kB 9.4 MB/s eta 0:00:00
# Downloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (25 kB)
# Downloading mpmath-1.3.0-py3-none-any.whl (536 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 536.2/536.2 kB 26.3 MB/s eta 0:00:00
# Downloading pytz-2024.1-py2.py3-none-any.whl (505 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 505.5/505.5 kB 24.9 MB/s eta 0:00:00
# Downloading tqdm-4.66.4-py3-none-any.whl (78 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 78.3/78.3 kB 10.8 MB/s eta 0:00:00
# Downloading tzdata-2024.1-py2.py3-none-any.whl (345 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 345.4/345.4 kB 18.6 MB/s eta 0:00:00
# Downloading nvidia_nvjitlink_cu12-12.5.82-py3-none-manylinux2014_x86_64.whl (21.3 MB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.3/21.3 MB 32.9 MB/s eta 0:00:00
# Downloading requests-2.32.3-py3-none-any.whl (64 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 64.9/64.9 kB 7.1 MB/s eta 0:00:00
# Downloading certifi-2024.7.4-py3-none-any.whl (162 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 163.0/163.0 kB 17.1 MB/s eta 0:00:00
# Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (142 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 142.1/142.1 kB 17.0 MB/s eta 0:00:00
# Downloading idna-3.7-py3-none-any.whl (66 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 66.8/66.8 kB 9.4 MB/s eta 0:00:00
# Downloading urllib3-2.2.2-py3-none-any.whl (121 kB)
#    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 121.4/121.4 kB 16.1 MB/s eta 0:00:00
# Installing collected packages: torchsummary, pytz, mpmath, flatbuffers, urllib3, tzdata, typing-extensions, traitlets, tqdm, threadpoolctl, tensorboard-data-server, sympy, six, safetensors, pyyaml, pyparsing, protobuf, pillow, packaging, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, MarkupSafe, markdown, kiwisolver, joblib, itsdangerous, idna, humanfriendly, grpcio, fsspec, fonttools, filelock, cycler, click, charset-normalizer, certifi, blinker, absl-py, Werkzeug, triton, scipy, requests, python-dateutil, opencv-python, onnx, nvidia-cusparse-cu12, nvidia-cudnn-cu12, ml-dtypes, matplotlib-inline, Jinja2, contourpy, coloredlogs, tensorboard, scikit-learn, pandas, onnxscript, onnxruntime, nvidia-cusolver-cu12, matplotlib, huggingface_hub, Flask, torch, seaborn, torchvision, timm
# Successfully installed Flask-3.0.3 Jinja2-3.1.4 MarkupSafe-2.1.5 Werkzeug-3.0.3 absl-py-2.1.0 blinker-1.8.2 certifi-2024.7.4 charset-normalizer-3.3.2 click-8.1.7 coloredlogs-15.0.1 contourpy-1.2.1 cycler-0.12.1 filelock-3.15.4 flatbuffers-24.3.25 fonttools-4.53.1 fsspec-2024.6.1 grpcio-1.64.1 huggingface_hub-0.23.4 humanfriendly-10.0 idna-3.7 itsdangerous-2.2.0 joblib-1.4.2 kiwisolver-1.4.5 markdown-3.6 matplotlib-3.9.0 matplotlib-inline-0.1.7 ml-dtypes-0.4.0 mpmath-1.3.0 networkx-3.3 numpy-1.26.4 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.20.5 nvidia-nvjitlink-cu12-12.5.82 nvidia-nvtx-cu12-12.1.105 onnx-1.16.1 onnxruntime-1.18.1 onnxscript-0.1.0.dev20240710 opencv-python-4.10.0.84 packaging-24.1 pandas-2.2.2 pillow-10.3.0 protobuf-5.27.2 pyparsing-3.1.2 python-dateutil-2.9.0.post0 pytz-2024.1 pyyaml-6.0.1 requests-2.32.3 safetensors-0.4.3 scikit-learn-1.5.0 scipy-1.13.1 seaborn-0.13.2 six-1.16.0 sympy-1.13.0 tensorboard-2.16.2 tensorboard-data-server-0.7.2 threadpoolctl-3.5.0 timm-1.0.7 torch-2.3.1 torchsummary-1.5.1 torchvision-0.18.1 tqdm-4.66.4 traitlets-5.14.3 triton-2.3.1 typing-extensions-4.12.2 tzdata-2024.1 urllib3-2.2.2
# WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
#  ---> Removed intermediate container 1d4d6bd4812c
#  ---> 5f5cdcbfb938
# Step 6/7 : EXPOSE 5000
#  ---> Running in 74dcc2ff8f57
#  ---> Removed intermediate container 74dcc2ff8f57
#  ---> e9832f58e89c
# Step 7/7 : CMD ["python", "app/ThyroidCancerClassifier/src/gui/app.py"]
#  ---> Running in ea8194c3f1f3
#  ---> Removed intermediate container ea8194c3f1f3
#  ---> 6eb4bbd80dde
# Successfully built 6eb4bbd80dde
# Successfully tagged thyroidcancerclassifier:latest                                                                                                 0.0s

# $ docker login  -u harito97
# Enter password: or access token
# $ docker tag thyroidcancerclassifier:latest harito97/thyroidcancerclassifier:latest     
# $ docker push harito97/thyroidcancerclassifier:latest

# $ docker pull harito97/thyroidcancerclassifier:latest
# $ sudo systemctl start docker
# $ sudo docker run -d -p 5000:5000 harito97/thyroidcancerclassifier:latest