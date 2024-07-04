# from torch.utils.data import Dataset
# from PIL import Image
# import os

# class CustomImageDataset(Dataset):
#     def __init__(self, img_dir, transform=None):
#         self.img_dir = img_dir
#         self.transform = transform
#         self.img_names = os.listdir(img_dir)

#     def __len__(self):
#         return len(self.img_names)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_names[idx])
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image

# Cái này có thể dùng để tránh tràn RAM khi load ảnh