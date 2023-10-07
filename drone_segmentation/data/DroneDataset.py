from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
import cv2
import os
import torch
import albumentations as A
import timm
import numpy as np
import torchvision.transforms as T

class DroneDataset(Dataset):
    def __init__(self, root_folder: str, img_size: tuple, is_training: bool, context_img_size : tuple = None):
        self.img_size = img_size
        self.is_training = is_training
        self.image_dir = os.path.join(root_folder, "image")
        self.mask_dir = os.path.join(root_folder, "mask")
        self.image_paths = glob(self.image_dir + "/*.png")
        self.context_img_size = context_img_size
        self.select_big = A.Compose(
            [
                A.RandomResizedCrop(
                    height=img_size[0]*2,
                    width=img_size[1]*2,
                    scale=(0.5, 1),
                    ratio=(0.9, 1.1),
                    always_apply=True,
                ),
                A.RandomRotate90(p=1),
                A.Transpose(p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ]
        )
        self.select_small = A.CenterCrop(height=img_size[0], width=img_size[1])
    def __len__(self):
        return len(self.image_paths) 
    
    def resize(self, img, size):
        return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_name = img_path.split("/")[-1]
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        big_selection = self.select_big(image=img, mask=mask)
        big_image = big_selection['image']
        big_mask = big_selection['mask']
        
        small_selection = self.select_small(image=big_image, mask=big_mask)
        small_image = small_selection['image']
        small_mask = small_selection['mask']
                                      
        small_mask = (small_mask > 0).astype(np.uint8)
        small_mask = self.resize(small_mask, self.img_size) 
        big_image = self.resize(big_image, self.context_img_size)
        big_image = Image.fromarray(big_image)
        
        small_image = Image.fromarray(small_image)
        
        return big_image, small_image, small_mask
    

class Collator(object):
    def __init__(self, transform, is_train=False, visualize_dir = 'debug'):
        self.transform = transform
        self.is_train = is_train
        self.visualized = False
        self.visualize_dir = visualize_dir
        os.makedirs('debug', exist_ok=True)
        
    def __call__(self, batch):
        big_imgs = []
        small_imgs = []
        masks = []
        for big_image, small_image, mask in batch:
            big_imgs.append(big_image)
            small_imgs.append(small_image)
            masks.append(mask)
        if not self.visualized:
            for i, (big_img, small_img, mask) in enumerate(zip(big_imgs, small_imgs, masks)):
                big_img.save(os.path.join(self.visualize_dir, f"big_img_{i}.jpg"))
                small_img.save(os.path.join(self.visualize_dir, f"small_img_{i}.jpg"))
                cv2.imwrite(os.path.join(self.visualize_dir, f"mask_{i}.png"), mask*255)
            self.visualized = True
        big_imgs = [self.transform(img) for img in big_imgs]
        small_imgs = [self.transform(img) for img in small_imgs]
        masks = [torch.FloatTensor(mask).unsqueeze(0) for mask in masks]
        big_images = torch.stack(big_imgs, dim=0)
        small_images = torch.stack(small_imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        
        return big_images, small_images, masks
    
    
def get_transform(backbone_name):
    try:
        data_config = timm.get_pretrained_cfg(backbone_name).to_dict()
        mean = data_config["mean"]
        std = data_config["std"]
    except:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return transform

if __name__ == '__main__':
    ds = DroneDataset('/mnt/data/luantranthanh/pyramid_segmentation/prepare_data/dataset/train/', (512,512), True)
    print(ds[0])
    transforms = get_transform('resnet50')
    dl = DataLoader(ds, batch_size = 2, collate_fn = Collator(transforms))
    print(next(iter(dl)))