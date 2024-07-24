import os
import glob
import cv2
from PIL import Image

def main(dataver1_dir="/dataver1", dataver4_dir="/dataver4"):
    for subset_dir in ["train", "valid", "test"]:
        for label in ['.B2', 'B5', 'B6']:
            for index, image_path in enumerate(
                glob.glob(os.path.join(dataver1_dir, subset_dir, label, "*.jpg"))
            ):
                image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_np)

                # Resize ảnh đầu vào về kích thước 1024x768
                resized_image = image_pil.resize((1024, 768))
                width, height = resized_image.size

                # Kích thước ảnh đầu ra
                out_width, out_height = 256, 256

                # Tính toán số lượng ảnh cắt ra
                n_patches_x = width // out_width
                n_patches_y = height // out_height

                # Đảm bảo chỉ cắt ra 12 ảnh
                if n_patches_x * n_patches_y < 12:
                    raise ValueError("Ảnh đầu vào không đủ lớn để cắt thành 12 ảnh 256x256")

                # Lưu các ảnh đã chia vào dataver4
                save_dir = os.path.join(dataver4_dir, subset_dir, label)
                os.makedirs(save_dir, exist_ok=True)

                patch_index = 0
                for y in range(0, height - out_height + 1, out_height):
                    for x in range(0, width - out_width + 1, out_width):
                        if patch_index >= 12:
                            break
                        patch = resized_image.crop((x, y, x + out_width, y + out_height))
                        save_path = os.path.join(save_dir, f"{index}_{patch_index}.jpg")
                        patch.save(save_path)
                        patch_index += 1
                    if patch_index >= 12:
                        break

                print(f'Saved {patch_index} images cropped to {save_dir}')

if __name__ == "__main__":
    main()
