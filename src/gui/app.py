import io
import json
from PIL import Image
from flask import Flask, jsonify, request
import onnxruntime
import numpy as np
import torchvision.transforms as transforms

app = Flask(__name__)

imagenet_class_index = json.load(open("classes_B2_B5B6.json"))

# Đường dẫn đến mô hình ONNX
model_destination_path = (
    "/mnt/Data/Projects/best_h97_resnet_B2_B5B6_dataver1_model.onnx"
)

# Khởi tạo phiên suy luận
if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
    providers = ["CUDAExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]

ort_session = onnxruntime.InferenceSession(model_destination_path, providers=providers)

# Tên đầu vào của mô hình
input_name = ort_session.get_inputs()[0].name


def transform_image(image_bytes):
    my_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    # Chuyển tensor numpy để sử dụng trong onnxruntime
    tensor_numpy = tensor.numpy()
    outputs = ort_session.run(None, {input_name: tensor_numpy})
    # Lấy chỉ số của lớp dự đoán cao nhất
    predicted_idx = str(np.argmax(outputs[0]))
    class_id, class_name = imagenet_class_index[predicted_idx]
    return class_id, class_name


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({"class_id": class_id, "class_name": class_name})


if __name__ == "__main__":
    app.run()
