import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from drone_segmentation.model import UperNet, SegFormer, UperNet_HF, ContextNet
from torchmetrics import JaccardIndex
import glob
from PIL import Image
import cv2
import numpy as np
from clearml import Logger
from tqdm import tqdm
import os


class LiTSeg(pl.LightningModule):
    def __init__(self, cfg, total_steps, transforms, context_net=True):
        super().__init__()
        self.cfg = cfg
        self.transforms = transforms
        if context_net:
            self.model = ContextNet(cfg.model.backbone_name, cfg.model.backbone_context_name, cfg.model.upernet_cfg, cfg.model.pretrained)
        elif "segformer" in cfg.model.backbone_name:
            self.model = SegFormer(cfg.model.backbone_name)
        elif "upernet" in cfg.model.backbone_name:
            self.model = UperNet_HF(cfg.model.backbone_name)
        else:
            self.model = UperNet(num_classes=1, backbone_name=cfg.model.backbone_name)
        self.total_steps = total_steps
        self.iou_calc = JaccardIndex('binary')
        self.save_hyperparameters()
    
    def forward(self, x, context, use_interpolate=True):
        N, C, H, W = x.shape
        logits = self.model(x, context)
        if use_interpolate:
            logits = F.interpolate(logits, (H, W), mode="bilinear", align_corners=False)
        return logits
    
    def training_step(self, batch, batch_id):
        big_imgs, small_imgs, masks = batch
        logits = self(small_imgs, big_imgs)
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_id):
        big_imgs, small_imgs, masks = batch
        logits = self(small_imgs, big_imgs)
        loss = F.binary_cross_entropy_with_logits(logits, masks)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        mask_prediction = logits.sigmoid()
        self.iou_calc.update(mask_prediction, masks)
    
    def on_validation_epoch_end(self):
        iou_score = self.iou_calc.compute()
        self.iou_calc.reset()
        self.log("val_iou", iou_score, on_epoch=True, prog_bar=True)
        # predict_masks, original_imgs, img_paths = self.predict_folder(self.cfg.data.test_folder)
        # for i, predict_mask in enumerate(predict_masks):
        #     img_name = os.path.basename(img_paths[i])
        #     original_img = original_imgs[i]
        #     color = np.array([0, 255, 0], dtype="uint8")
        #     masked_img = np.where(predict_mask[..., None], color, original_img)
        #     colored_mask_img = cv2.addWeighted(original_img, 0.6, masked_img, 0.4, 0)
        #     Logger.current_logger().report_image(
        #         "Testing Image",
        #         img_name,
        #         iteration=self.trainer.current_epoch,
        #         image=colored_mask_img,
        #     )
        
    def predict_a_image(self, image_path):
        img_name = os.path.basename(image_path)
        img = cv2.imread(image_path)
        W, H, C = img.shape
        IMG_SIZE = self.cfg.data.img_size[0]
        IMG_SIZE = 1024
        STRIDE = IMG_SIZE
        logits_map = np.zeros((W, H))
        for top_left_y in range(0, H, STRIDE):
            for top_left_x in range(0, W, STRIDE):
                # Calculate the bottom right coordinates of the sub-image
                bottom_right_x = min(top_left_x + IMG_SIZE, W)
                bottom_right_y = min(top_left_y + IMG_SIZE, H)
                # Extract the sub-image
                sub_img = img[top_left_x:bottom_right_x, top_left_y:bottom_right_y]
                sub_img = Image.fromarray(sub_img)
                sub_img = self.transforms(sub_img).unsqueeze(0).to(self.device)
                # Process the sub-image and update the logits map
                sub_logits = self(sub_img).squeeze(0).squeeze(0)
                sub_logits = sub_logits.cpu().numpy()
                logits_map[top_left_x:top_left_x+IMG_SIZE, top_left_y:top_left_y+IMG_SIZE] = sub_logits
        predict_mask = logits_map > 0
        predict_mask = np.array(predict_mask, dtype=np.uint8)*255
        return predict_mask, img
    @torch.inference_mode()
    def predict_folder(self, image_dir):
        img_paths = glob.glob(image_dir + '/*.tif')[:2]
        predict_masks = []
        original_imgs = []
        for img_path in tqdm(img_paths):
            predict_mask, img = self.predict_a_image(img_path)
            predict_masks.append(predict_mask)
            original_imgs.append(img)
        return predict_masks, original_imgs, img_paths
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            weight_decay=self.cfg.training.weight_decay,
            lr=float(self.cfg.training.lr),
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(self.cfg.training.lr),
                    total_steps=self.total_steps,
                    pct_start=0.1,
                    div_factor=25,
                    final_div_factor=1e4,
                ),
                "interval": "step",
            },
        }
        