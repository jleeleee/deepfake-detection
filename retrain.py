import os
from glob import glob

import fire
import lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import torch
import torch.nn as nn
import yaml
from lightning import Trainer
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from rich import traceback

from src import model as models
from src.config import Backbone, Config, Head, load_config
from src.utils import files
from src.utils.checks import checks
from src.utils.model_checkpoint import ModelCheckpointParallel
from sklearn.model_selection import train_test_split
from customdata import ImageDataset, DeepfakeDataModule
traceback.install()
# Define the root directory of the dataset
dataset_root = '/scratch/jlee436/cs584/data/'
# Load the train CSV file
train_df = pd.read_csv(os.path.join(dataset_root, 'train.csv'), index_col=0)
# Load the test CSV file
# test_df = pd.read_csv(os.path.join(dataset_root, 'test.csv'))

train_df.head()

# Split into training and validation (70% train, 30% test)
train_val_df, test_df = train_test_split(train_df, test_size=0.3, random_state=42, stratify=train_df['label'])

# further split train_val_df into train and validation (80% train, 20% validation)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label'])

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

# class ImageDataset(Dataset):
#     def __init__(self, df, root_dir, transform=None, is_test=False):
#         self.df = df
#         self.root_dir = root_dir
#         self.transform = transform
#         self.is_test = is_test  # Flag to indicate if this is the test dataset

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         if self.is_test:
#             # Use the first column (assumed to be 'id' or 'file_name')
#             img_path = os.path.join(self.root_dir, self.df.iloc[idx, 0])  
#         else:
#             # Use column names instead of hardcoded index
#             img_path = os.path.join(self.root_dir, self.df['file_name'].iloc[idx])  
#             label = int(self.df['label'].iloc[idx])  

#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         if self.is_test:
#             return image, -1 
#         else:
#             return image, label

def main(config: Config, train: bool):
    checks(config)

    torch.set_float32_matmul_precision("high")  # Set the precision for matmul operations

    model = models.DeepfakeDetectionModel(config, verbose=True)

    if config.checkpoint:
        model.load_state_dict(torch.load(config.checkpoint, map_location="cpu", weights_only=True)["state_dict"])

    data_module = DeepfakeDataModule(config, model.get_preprocessing())

    loggers: list = [pl_loggers.CSVLogger(config.run_dir, name=config.run_name, version="")]

    if config.wandb:
        wandb_logger = pl_loggers.WandbLogger(
            project="deepfake",
            name=config.run_name,
            save_dir=f"{config.run_dir}/{config.run_name}",
            tags=config.wandb_tags,
        )
        loggers.append(wandb_logger)

    callbacks = [
        pl_callbacks.RichProgressBar(),
        ModelCheckpointParallel(filename="best_mAP", monitor="val/mAP_video", mode="max"),
    ]

    trainer = Trainer(
        devices=config.devices,
        max_epochs=config.max_epochs,
        precision=config.precision,
        accumulate_grad_batches=config.batch_size // config.mini_batch_size,
        fast_dev_run=config.fast_dev_run,
        log_every_n_steps=100,
        overfit_batches=config.overfit_batches,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=config.limit_test_batches,
        deterministic=config.deterministic,
        detect_anomaly=config.detect_anomaly,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=config.run_dir,
    )

    if train:
        trainer.fit(model, data_module)

        ckpt_path = f"{config.run_dir}/{config.run_name}/checkpoints/{config.checkpoint_for_testing}.ckpt"
        trainer.test(model, data_module, ckpt_path=ckpt_path)

    else:
        assert config.checkpoint is not None, "Checkpoint is required for testing"
        trainer.test(model, data_module)

    if config.wandb:
        wandb_logger.finalize("success")
        wandb_logger.experiment.finish()


def get_train_config() -> Config:
    config = Config()

    config.run_name = "example-run"
    config.run_dir = "runs/train"
    config.wandb = False

    config.num_workers = 24
    config.devices = [0]

    config.backbone = Backbone.CLIP_L_14
    config.freeze_feature_extractor = True
    config.peft.enabled = True
    config.peft.ln_tuning.enabled = True
    config.head = Head.LinearNorm
    config.num_classes = 2
    config.loss.ce_labels = 1.0
    config.slerp_feature_augmentation = True

    config.batch_size = config.mini_batch_size = 128
    config.lr_scheduler = "cosine"
    config.lr = 8e-5
    config.min_lr = 5e-5
    config.weight_decay = 0
    config.max_epochs = 10

    limit_val_files = 16384
    config.limit_val_files = limit_val_files
    config.limit_val_batches = limit_val_files // config.mini_batch_size

    config.binary_labels = True

    # df_trn = pd.read_csv("train_split.csv")
    # df_val = pd.read_csv("val_split.csv")
    # base_image_dir = "/scratch/jlee436/cs584/data/"
    # df_trn["file_name"] = df_trn["file_name"].apply(lambda x: os.path.join(base_image_dir, x))
    # df_val["file_name"] = df_val["file_name"].apply(lambda x: os.path.join(base_image_dir, x))
    # config.trn_files = df_trn["file_name"].tolist()
    # config.val_files = df_val["file_name"].tolist()

    # config.trn_files = [
    #     "config/datasets/FF/test/DF.txt",
    #     "config/datasets/FF/test/F2F.txt",
    #     "config/datasets/FF/test/FS.txt",
    #     "config/datasets/FF/test/NT.txt",
    #     "config/datasets/FF/test/real.txt",
    # ]
    # config.val_files = [
    #     "config/datasets/CDFv2/test/Celeb-synthesis.txt",
    #     "config/datasets/CDFv2/test/Celeb-real.txt",
    #     "config/datasets/CDFv2/test/YouTube-real.txt",
    # ]

    # config.tst_files = {
    #     "CDF": [
    #         "config/datasets/CDFv2/test/Celeb-synthesis.txt",
    #         "config/datasets/CDFv2/test/Celeb-real.txt",
    #         "config/datasets/CDFv2/test/YouTube-real.txt",
    #     ]
    

    return config


def get_test_config() -> Config:
    config_path = "runs/train/example-run/hparams.yaml"
    new_run_name = "example-run"

    config = load_config(config_path)

    config.run_name = new_run_name
    config.run_dir = "runs/test"
    config.checkpoint = config_path.replace("hparams.yaml", "checkpoints/best_mAP.ckpt")
    config.wandb = False
    config.wandb_tags.extend(["test"])

    config.num_workers = 12
    config.batch_size = config.mini_batch_size = 512
    config.devices = [0]

    # config.tst_files = {
    #     "CDF": [
    #         "config/datasets/CDFv2/test/Celeb-synthesis.txt",
    #         "config/datasets/CDFv2/test/Celeb-real.txt",
    #         "config/datasets/CDFv2/test/YouTube-real.txt",
    #     ]
    # }

    return config


def get_debug_config(config: Config) -> Config:
    #! Debug

    config.run_name = "tmp"

    config.devices = [0]

    config.num_workers = 8
    # config.batch_size = config.mini_batch_size = 512
    config.max_epochs = 1
    config.limit_train_batches = 12
    config.limit_val_batches = 12
    config.limit_test_batches = 12
    config.deterministic = True
    config.detect_anomaly = True

    return config


def entry(train: bool = False, test: bool = False, debug: bool = False, **kwargs):
    if train:
        config = get_train_config()

    elif test:
        config = get_test_config()

    else:
        raise ValueError("Either --train or --test must be provided")

    # Overwrite config with debug values
    if debug:
        config = config.model_copy(update=dict(get_debug_config(config)))

    # Parse command line arguments
    config = config.model_copy(update=kwargs)

    # Revalidate the config - checks if user provided valid values
    config = Config(**dict(config))

    main(config, train)


if __name__ == "__main__":
    fire.Fire(entry)
