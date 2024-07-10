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
# [+] Building 259.1s (8/9)                                                                                                                              docker:default
#  => [internal] load build definition from Dockerfile                                                                                                             0.1s
#  => => transferring dockerfile: 830B                                                                                                                             0.0s
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s
#  => [internal] load .dockerignore                                                                                                                                0.0s
#  => => transferring context: 2B                                                                                                                                  0.0s
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8s
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                                                                 0.9s
#  => => sha256:726c326e9564077eaab0c9736248c43b9a576d8e80226926ed05e7bc21f21c20 230B / 230B                                                                       1.0s
#  => => extracting sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc                          [+] Building 259.3s (8/9)                                                                                                                              docker:default1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load build definition from Dockerfile                                                                                                             0.1s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
#  => [internal] load .dockerignore                                                                                                                                0.0sgithub.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCance
#  => => transferring context: 2B                                                                                                                                  0.0s                                                                 
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8s                                                                 
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s                                                                 
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s                                                                 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s                                                                 
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s                                                                 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s                                                                 
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                   [+] Building 259.4s (8/9)                                                                                                                              docker:default6d8e80226926ed05e7bc21f21c20 230B / 230B                         
#  => [internal] load build definition from Dockerfile                                                                                                             0.1sda21092098e5ca72be6ac4703e1fb51ae1a15bc                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => [internal] load .dockerignore                                                                                                                                0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => => transferring context: 2B                                                                                                                                  0.0s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8sgithub.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCance
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s                                                                 
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s                                                                 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s                                                                 
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s                                                                 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s                                                                 
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                   [+] Building 259.6s (8/9)                                                                                                                              docker:default6d8e80226926ed05e7bc21f21c20 230B / 230B                         
#  => [internal] load build definition from Dockerfile                                                                                                             0.1sda21092098e5ca72be6ac4703e1fb51ae1a15bc                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => [internal] load .dockerignore                                                                                                                                0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => => transferring context: 2B                                                                                                                                  0.0s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8sgithub.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCance
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s                                                                 
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s                                                                 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s                                                                 
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s                                                                 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s                                                                 
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                   [+] Building 259.7s (8/9)                                                                                                                              docker:default6d8e80226926ed05e7bc21f21c20 230B / 230B                         
#  => [internal] load build definition from Dockerfile                                                                                                             0.1sda21092098e5ca72be6ac4703e1fb51ae1a15bc                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => [internal] load .dockerignore                                                                                                                                0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => => transferring context: 2B                                                                                                                                  0.0s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8sgithub.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCance
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s                                                                 
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s                                                                 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s                                                                 
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s                                                                 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s                                                                 
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                   [+] Building 259.9s (8/9)                                                                                                                              docker:default6d8e80226926ed05e7bc21f21c20 230B / 230B                         
#  => [internal] load build definition from Dockerfile                                                                                                             0.1sda21092098e5ca72be6ac4703e1fb51ae1a15bc                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => [internal] load .dockerignore                                                                                                                                0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => => transferring context: 2B                                                                                                                                  0.0s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8sgithub.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCance
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s                                                                 
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s                                                                 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s                                                                 
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s                                                                 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s                                                                 
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                   [+] Building 260.0s (8/9)                                                                                                                              docker:default6d8e80226926ed05e7bc21f21c20 230B / 230B                         
#  => [internal] load build definition from Dockerfile                                                                                                             0.1sda21092098e5ca72be6ac4703e1fb51ae1a15bc                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => [internal] load .dockerignore                                                                                                                                0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => => transferring context: 2B                                                                                                                                  0.0s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8sgithub.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCance
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s                                                                 
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s                                                                 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s                                                                 
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s                                                                 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s                                                                 
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                   [+] Building 260.2s (8/9)                                                                                                                              docker:default6d8e80226926ed05e7bc21f21c20 230B / 230B                         
#  => [internal] load build definition from Dockerfile                                                                                                             0.1sda21092098e5ca72be6ac4703e1fb51ae1a15bc                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => [internal] load .dockerignore                                                                                                                                0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => => transferring context: 2B                                                                                                                                  0.0s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8sgithub.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCance
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s                                                                 
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s                                                                 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s                                                                 
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s                                                                 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s                                                                 
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                   [+] Building 260.3s (8/9)                                                                                                                              docker:default6d8e80226926ed05e7bc21f21c20 230B / 230B                         
#  => [internal] load build definition from Dockerfile                                                                                                             0.1sda21092098e5ca72be6ac4703e1fb51ae1a15bc                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => [internal] load .dockerignore                                                                                                                                0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => => transferring context: 2B                                                                                                                                  0.0s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8sgithub.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCance
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s                                                                 
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s                                                                 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s                                                                 
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s                                                                 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s                                                                 
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                   [+] Building 260.5s (8/9)                                                                                                                              docker:default6d8e80226926ed05e7bc21f21c20 230B / 230B                         
#  => [internal] load build definition from Dockerfile                                                                                                             0.1sda21092098e5ca72be6ac4703e1fb51ae1a15bc                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => [internal] load .dockerignore                                                                                                                                0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => => transferring context: 2B                                                                                                                                  0.0s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8sgithub.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCance
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s                                                                 
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s                                                                 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s                                                                 
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s                                                                 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s                                                                 
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                   [+] Building 260.6s (8/9)                                                                                                                              docker:default6d8e80226926ed05e7bc21f21c20 230B / 230B                         
#  => [internal] load build definition from Dockerfile                                                                                                             0.1sda21092098e5ca72be6ac4703e1fb51ae1a15bc                          
#  => => transferring dockerfile: 830B                                                                                                                             0.0s1cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                     
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s6df5f4992c0e5148b71fc4989f228df99f96342                          
#  => [internal] load .dockerignore                                                                                                                                0.0s248c43b9a576d8e80226926ed05e7bc21f21c20                          
#  => => transferring context: 2B                                                                                                                                  0.0s24c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                          
# [+] Building 325.3s (9/9) FINISHED                                                                                                                     docker:defaultg
#  => [internal] load build definition from Dockerfile                                                                                                             0.1s
#  => => transferring dockerfile: 830B                                                                                                                             0.0s 
#  => [internal] load metadata for docker.io/library/python:3.10                                                                                                   1.9s
#  => [internal] load .dockerignore                                                                                                                                0.0s 
#  => => transferring context: 2B                                                                                                                                  0.0s
#  => [1/5] FROM docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             3.8s 
#  => => resolve docker.io/library/python:3.10@sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2                                             0.1s
#  => => sha256:2ca0144d159a98c1211520b6a1235fb0f312c842c3d0bbda0ec4f930b8d596a5 2.52kB / 2.52kB                                                                   0.0s 
#  => => sha256:506eee363017f0b9c7f06f4839e7db90d1001094882e8cff08c8261ba2e05be2 9.08kB / 9.08kB                                                                   0.0s
#  => => sha256:6905cae1f812b8c17ad0a8c5a4690420aaf1f7fc8834dd1a9b555010ffe1dac0 7.28kB / 7.28kB                                                                   0.0s 
#  => => sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc 6.16MB / 6.16MB                                                                   0.5s
#  => => sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342 17.15MB / 17.15MB                                                                 0.9s
#  => => sha256:726c326e9564077eaab0c9736248c43b9a576d8e80226926ed05e7bc21f21c20 230B / 230B                                                                       1.0s
#  => => extracting sha256:fd155188cf05f7056f1877236da21092098e5ca72be6ac4703e1fb51ae1a15bc                                                                        0.9s
#  => => sha256:331871c631e854245e4bca24124c51b15e451cd3bbeafe252d23b1c1b2c0ebc7 3.08MB / 3.08MB                                                                   1.0s
#  => => extracting sha256:f0d61d0facb63beecd35be45a6df5f4992c0e5148b71fc4989f228df99f96342                                                                        0.6s
#  => => extracting sha256:726c326e9564077eaab0c9736248c43b9a576d8e80226926ed05e7bc21f21c20                                                                        0.0s
#  => => extracting sha256:331871c631e854245e4bca24124c51b15e451cd3bbeafe252d23b1c1b2c0ebc7                                                                        0.3s
#  => [2/5] RUN mkdir /app &&     git clone https://github.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCancerClassifier                                   2.6s
#  => [3/5] WORKDIR /app/ThyroidCancerClassifier                                                                                                                   0.3s
#  => [4/5] RUN pip install --upgrade pip                                                                                                                          3.5s
#  => [5/5] RUN pip install -r requirements.txt                                                                                                                  202.8s
#  => exporting to image                                                                                                                                         109.8s
#  => => exporting layers                                                                                                                                        109.6s
#  => => writing image sha256:a967fcb128206a69cbbb3415e3e44f98fcbd2423085e04c4ad00d58b26532c5a                                                                     0.0s
#  => => naming to docker.io/library/thyroidcancerclassifier                                                                                                       0.0s

# $ docker login  -u harito97
# Enter password: or access token
# $ docker tag thyroidcancerclassifier:latest harito97/thyroidcancerclassifier:latest     
# $ docker push harito97/thyroidcancerclassifier:latest