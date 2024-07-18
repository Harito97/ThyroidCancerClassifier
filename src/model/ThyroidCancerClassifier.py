import torch
from ultralytics import YOLO
from src.model.classifier.H97 import H97_EfficientNet
from src.data_preparation.ThyroidCancerDataset import ThyroidCancerDataset
import torchvision.transforms as transforms
import os
import glob
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random


class ThyroidCancerClassifier:
    def __init__(self, ICCD_model_path, BTTC_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ICCD_model = YOLO(ICCD_model_path)
        self.BTTC_model = H97_EfficientNet()
        self.BTTC_model.load_state_dict(
            torch.load(BTTC_model_path, map_location=self.device)
        )
        self.BTTC_model.eval()
        self.BTTC_model.to(self.device)
        self.transform = transforms.Compose(
            [
                # transforms.Resize((256, 256)),
                # transforms.RandomCrop(224),
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
        self.transform2 = transforms.Compose([transforms.ToTensor()])
        torch.no_grad()

    def predict(self, image_path):
        best_images = self.step1(image_path)
        results = self.step2(best_images)
        label = self.step3(results)
        return label

    def step1(self, image_path):
        """
        Crop 5 best images from 1 origin image
        """
        # Load image
        image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_np)

        # Identify bounding boxes
        ICCD_bounding_boxes = self.__identify_bounding_boxes(
            self.ICCD_model, image_path
        )
        print(type(ICCD_bounding_boxes))
        # Remove noise
        _, areas, bounding_boxes_list = self.__remove_noise(
            ICCD_bounding_boxes, image_np
        )  # _ is denoised_image

        # Get 5 largest areas
        return self.__crop_5_largest_areas(
            image_pil, areas, bounding_boxes_list, self.transform
        )

    def step2(self, images):
        """
        Predict the images
        """
        results = []
        for image in images:
            # Transform the image to a tensor
            image = self.transform2(image)
            image = image.unsqueeze(0)
            image = image.to(self.device)
            outputs = self.BTTC_model(image)
            _, preds = torch.max(outputs, 1)
            results.extend(preds.view(-1).cpu().numpy())
        return results

    def step3(self, results):
        """
        Return the most common label
        """
        return max(set(results), key=results.count)

    def __identify_bounding_boxes(self, model, image_path):
        """
        Identify bounding boxes of cell clusters in an image.

        Args:
            model (YOLO): Loaded YOLO model.
            image_path (str): Path to the input image.

        Returns:
            torch.Tensor: Tensor of bounding boxes in [x1, y1, x2, y2] format.
        """
        result = model(image_path)
        bounding_boxes = result[0].boxes.xyxy
        return bounding_boxes

    def __remove_noise(self, bounding_boxes, image_np):
        """
        Remove noise from the image by keeping only the areas within bounding boxes.

        Args:
            bounding_boxes (torch.Tensor): Tensor of bounding boxes.
            image_np (numpy.ndarray): Input image as a numpy array.

        Returns:
            tuple: (PIL.Image, dict, list) Denoised image, areas of bounding boxes, and bounding boxes as list.
        """
        areas = {}
        new_image = np.ones_like(image_np) * 255
        bounding_boxes_list = []

        for index, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = map(int, box)
            areas[index] = (x2 - x1) * (y2 - y1)
            bounding_boxes_list.append((x1, y1, x2, y2))
            # print(x1, y1, x2, y2)
            new_image[y1:y2, x1:x2] = image_np[y1:y2, x1:x2]

        if len(areas) > 0:
            return Image.fromarray(new_image), areas, bounding_boxes_list
        return Image.fromarray(image_np), areas, bounding_boxes_list

    def __get_5_largest_areas(self, areas):
        """
        Get the indices of the 5 largest areas.

        Args:
            areas (dict): Dictionary of area indices and their sizes.

        Returns:
            list: Indices of the 5 largest areas (or fewer if there are less than 5).
        """
        sorted_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
        return [index for index, _ in sorted_areas[:5]]

    def __augment_images(self, images, num_augmentations, augmentation_transforms):
        """
        Augment the given images to reach a total of 5 images using various augmentation techniques.

        Args:
            images (list): List of PIL.Image objects.
            num_augmentations (int): Number of augmentations needed.

        Returns:
            list: List of augmented PIL.Image objects.
        """
        augmented = images.copy()

        while len(augmented) < 5:
            # Randomly select an image to augment
            base_image = random.choice(images)

            # Apply the basic augmentation transforms
            augmented_image = augmentation_transforms(base_image)

            augmented.append(augmented_image)

        return augmented[:5]  # Ensure we return exactly 5 images

    def __crop_5_largest_areas(
        self, image, areas, bounding_boxes, augmentation_transforms
    ):
        """
        Crop the 5 largest areas from the image or create 5 sub-images if no bounding boxes.

        Args:
            image (PIL.Image): Input image.
            areas (dict): Dictionary of area indices and their sizes.
            bounding_boxes (list): List of bounding boxes.
            path_to_save (str): Path to save the cropped images (not used in this function).

        Returns:
            list: List of 5 PIL.Image objects (cropped or sub-images).
        """
        images = []
        if len(areas) > 0:
            for index in self.__get_5_largest_areas(areas):
                x1, y1, x2, y2 = bounding_boxes[index]
                cropped_image = image.crop((x1, y1, x2, y2))
                images.append(cropped_image)

            # Augment the image to have 5 areas
            if len(images) < 5:
                images = self.__augment_images(
                    images, 5 - len(images), augmentation_transforms
                )
        else:
            image_size = image.size
            x1, y1, x2, y2 = 0, 0, image_size[0] // 2, image_size[1] // 2
            images = [
                image.crop((x1, y1, x2, y2)),
                image.crop((x2, y1, image_size[0], y2)),
                image.crop((x1, y2, x2, image_size[1])),
                image.crop((x2, y2, image_size[0], image_size[1])),
                image,
            ]
        return images
