from torch.utils.data import DataLoader
from src.data_preparation.ThyroidCancerDataset import ThyroidCancerDataset

class ThyroidCancerDataLoader():
    def __init__(self, img_dir=None, transform=None, balance=True):
        self.img_dir = img_dir
        self.transform = transform
        self.balance = balance

    def get_data_loader(self, img_dir, classes={0: ["B2"], 1: ["B5", "B6"]}, mode='train', batch_size=32, num_workers=4):
        self.img_dir = img_dir
        dataset = ThyroidCancerDataset(img_dir=self.img_dir, classes={0: ["B2"], 1: ["B5", "B6"]}, transform=self.transform, balance=self.balance, mode=mode)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=num_workers)
        return data_loader