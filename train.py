from drone_segmentation.data import DroneDataset, Collator, get_transform
from drone_segmentation import LiTSeg
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml
import argparse
from easydict import EasyDict
from clearml import Task

torch.set_float32_matmul_precision('high')
pl.seed_everything(42)

def make_callbacks():
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor="val_iou",
            mode="max",
            save_top_k=1,
            save_last=True,
            filename="{epoch}-{val_iou:.4f}-{val_loss:.2f}",
        )
    )
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step"))
    return callbacks

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()

    return args

def prepare_data(cfg, args):
    train_dataset = DroneDataset(cfg.data.train_root_folder, cfg.data.img_size, True, cfg.data.context_img_size)
    val_dataset = DroneDataset(cfg.data.val_root_folder, cfg.data.img_size, False, cfg.data.context_img_size)
    print("Total Train Dataset:", len(train_dataset))
    print("Total Val Dataset:", len(val_dataset))
    transforms = get_transform(cfg.model.backbone_name)
    print(transforms)
    collator = Collator(transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=collator)
    return train_dataloader, val_dataloader, transforms

if __name__ == '__main__':
    
    args = get_args()
    
    with open(args.cfg_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = EasyDict(cfg)
        cfg.data.img_size = eval(cfg.data.img_size)
        cfg.data.context_img_size = eval(cfg.data.context_img_size)
    task = Task.init(project_name='drone_segmentation', task_name=cfg.model.backbone_name)
    task.set_parameters(cfg)    
    train_dataloader, val_dataloader, transforms = prepare_data(cfg, args)
    
    lit_seg = LiTSeg(cfg, len(train_dataloader)*args.epochs, transforms)
    logger = pl.loggers.TensorBoardLogger(
        save_dir="tensorboard_logs",
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.epochs,
        accelerator="auto",
        precision=32,
        callbacks=make_callbacks(),
        val_check_interval=0.75,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
    )

    trainer.fit(lit_seg, train_dataloader, val_dataloader)