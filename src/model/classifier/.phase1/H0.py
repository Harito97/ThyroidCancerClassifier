import torch
import os
import matplotlib.pyplot as plt
from math import ceil, sqrt


class H0(torch.nn.Module):
    def __init__(self):
        """
        Input đầu vào của mô hình là ảnh RGB kích thước 224x224
        """
        super(H0, self).__init__()

    def save_model(self, folder_path, file_name, type=".pt"):
        # Kiểm tra và chỉ thêm type nếu file_name chưa có định dạng
        if not file_name.endswith(".pt") and not file_name.endswith(".pth"):
            file_name = file_name.split(".")[0]  # Xóa phần mở rộng nếu có
            file_name += type
        file_path = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path, device="cpu"):
        # Thêm map_location để đảm bảo mô hình được tải đúng cách trên các thiết bị
        self.load_state_dict(torch.load(file_path, map_location=device))

    def get_feature_maps(self, x):
        # Phương thức này cần được ghi đè trong mỗi class con
        # để trả về bản đồ đặc trưng tương ứng với mô hình đó
        raise NotImplementedError("This method should be implemented by subclasses.")

    def visualize_feature_maps(self, input_image, show_last_layer=False):
        feature_maps = self.get_feature_maps(input_image)
        if show_last_layer:
            # Vẽ bản đồ đặc trưng của lớp cuối cùng
            plt.figure(figsize=(20, 15))
            # Tính số dòng và cột cho lưới hiển thị
            f_map = feature_maps[-1]
            n_channels = f_map.size(1)  # Số lượng kênh
            n_cols = int(ceil(sqrt(n_channels)))  # Số cột
            n_rows = int(ceil(n_channels / n_cols))  # Số dòng

            # Tạo lưới hiển thị các kênh
            for i in range(1, n_channels + 1):
                plt.subplot(n_rows, n_cols, i)
                plt.imshow(f_map[0, i - 1].cpu().detach().numpy(), cmap="gray")
                plt.axis("off")
            plt.show()
            return

        # Vẽ bản đồ đặc trưng của tất cả các lớp bản đồ đặc trưng rút ra được từ get_feature_maps
        for layer, f_map in enumerate(feature_maps):
            plt.figure(figsize=(20, 15))
            layer += 1
            # Giả sử mỗi bản đồ đặc trưng có n_channels
            n_channels = f_map.size(1)
            for i in range(1, n_channels + 1):
                plt.subplot(1, n_channels, i)
                plt.imshow(f_map[0, i - 1].cpu().detach().numpy(), cmap="gray")
                plt.axis("off")
            plt.show()
