# Sử dụng hình ảnh cơ sở Python 3.11
FROM python:3.11

# Cài đặt các gói cần thiết từ
RUN mkdir /app
RUN cd /app
RUN git clone https://github.com/Harito97/ThyroidCancerClassifier.git
WORKDIR /app/ThyroidCancerClassifier
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng 5000 cho ứng dụng Flask
EXPOSE 5000

# Chạy ứng dụng Flask khi khởi động container
CMD ["python", "app.py"]