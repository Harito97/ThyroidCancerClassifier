# Sử dụng hình ảnh cơ sở Python 3.11
FROM python:3.11

# Tạo thư mục làm việc và clone dự án, sau đó cài đặt các gói cần thiết
RUN mkdir /app && \
    git clone https://github.com/Harito97/ThyroidCancerClassifier.git /app/ThyroidCancerClassifier

WORKDIR /app/ThyroidCancerClassifier

RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng 5000 cho ứng dụng Flask
EXPOSE 5000

# Chạy ứng dụng Flask khi khởi động container
CMD ["python", "app.py"]