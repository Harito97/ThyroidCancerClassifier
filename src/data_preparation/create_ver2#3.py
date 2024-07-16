import os
import glob
from PIL import Image
from ultralytics import YOLO 
import cv2

def detect_cell_cluster(image_path):
    results = model(image_path)
    return results[0].boxes.xyxy.tolist()
    
# Step 1. Read image from train/valid/test folder + Identify the bounding boxes of the thyroid nodule cluster by cell cluster detection model

# Step 2.1. Remove the noise of images (not in the bounding boxes) and save them to the corresponding folder
# This will create dataver2

# Step 2.2. Crop the bounding boxes and save them to the corresponding folder
# this will create dataver3

def identify_bounding_boxes(model, image_path):
    result = model(image_path)  # results list    
    bounding_boxes = result[0].boxes.xyxy  # Giả sử kết quả trả về có dạng {'boxes': [[x1, y1, x2, y2], ...]}
    return bounding_boxes

def remove_noise(bounding_boxes, image_np):
    areas = {}
    new_image = np.ones_like(image_np) * 255
    for index, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = map(int, box)
        areas[index] = (x2 - x1) * (y2 - y1)
        bounding_boxes[index] = (x1, y1, x2, y2)
        print(x1, y1, x2, y2)
        new_image[y1:y2, x1:x2] = image_np[y1:y2, x1:x2]
    if len(areas) > 0:
        return Image.fromarray(new_image), areas, bounding_boxes
    return Image.fromarray(image_np), areas, bounding_boxes

def get_5_largest_areas(areas):
    ...  # return 5 largest areas or < 5 if there are less than 5 areas

def crop_5_largest_areas(image, areas, bounding_boxes, path_to_save):
    images = []
    if len(areas) > 0:
        for index in get_5_largest_areas(areas):
            x1, y1, x2, y2 = bounding_boxes[index]
            cropped_image = image.crop((x1, y1, x2, y2))
            images.append(cropped_image)
        # Augment the image to have 5 areas
        if len(images) < 5:
            images = augment_images(images, 5 - len(images))
            ... make real augment_images here (create the augment_images function)
    else:
        image_size = image.size
        x1, y1, x2, y2 = 0, 0, image_size[0] // 2, image_size[1] // 2
        images.append(image[y1:y2, x1:x2])
        images.append(image[y1:y2, x2:])
        images.append(image[y2:, x1:x2])
        images.append(image[y2:, x2:])
        images.append(image[:, :])
    return images
    

if __name__ == '__main__':
    model_path = '/Data/Projects/ThyroidCancerClassifier/src/model/cell_detect/runs/detect/train2/weights/best.pt' 
    model = YOLO(model_path)

    dataver1_dir = '/dataver1'
    dataver2_dir = '/dataver2'
    dataver3_dir = '/dataver3'

    for subset_dir in ['train', 'valid', 'test']:
        for index, image_path in enumerate(glob.glob(os.path.join(dataver1_dir, subset_dir, '*.jpg'))):
            image_np = cv2.imread(image_path)
            bounding_boxes = identify_bounding_boxes(model, image_path)
            print(f'{image_path} have {len(bounding_boxes)} cell importance clusters')
            
            image_dataver2, areas, bounding_boxes = remove_noise(bounding_boxes, image_np)
            save_dataver2_dir = os.path.join(dataver2_dir, subset_dir, f'{index}.jpg')
            image_dataver2.save(save_dataver2_dir)

            images_dataver3 = crop_5_largest_areas(image_np, areas, bounding_boxes, dataver2_dir)
            for i, img in enumerate(images_dataver3):
                # img = cv2.resize(img, (224, 224))
                save_dataver3_dir = os.path.join(dataver3_dir, subset_dir, f'{index}_{i}.jpg')
                img.save(save_dataver3_dir)