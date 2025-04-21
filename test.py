import os

import torch
from lightning.fabric import Fabric
from PIL import Image

from src.config import Config
from src.model.dfdet import DeepfakeDetectionModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

DEVICES = [0]

torch.set_float32_matmul_precision("high")

# Check if weights/model.ckpt exists, if not, download it from huggingface
model_path = "weights/model.ckpt"
if not os.path.exists(model_path):
    print("Downloading model")
    os.makedirs("weights", exist_ok=True)
    os.system(f"wget https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.ckpt -O {model_path}")

# Load checkpoint
ckpt = torch.load(model_path, map_location="cpu")

run_name = ckpt["hyper_parameters"]["run_name"]
print(run_name)

# Initialize model from config
model = DeepfakeDetectionModel(Config(**ckpt["hyper_parameters"]))
model.eval()

# Load model state dict
model.load_state_dict(ckpt["state_dict"])

# Get preprocessing function
preprocessing = model.get_preprocessing()

import pandas as pd
image_dir_base = "/scratch/jlee436/cs584/data/"
# Load 20 images
df_val = pd.read_csv("val_split.csv")
# paths = [os.path.join(image_dir_base, image) for image in df_val["file_name"].tolist()[:20]]


# # To pillow images
# pillow_images = [Image.open(image) for image in paths]

# # To tensors
# batch_images = torch.stack([preprocessing(image) for image in pillow_images])

precision = ckpt["hyper_parameters"]["precision"]
fabric = Fabric(accelerator="cuda", devices=DEVICES, precision=precision)
fabric.launch()
model = fabric.setup_module(model)

# # perform inference
# with torch.no_grad():
#     # Move batch_images to the correct device and dtype
#     batch_images = batch_images.to(fabric.device).to(model.dtype)

#     # Forward pass
#     output = model(batch_images)

# # logits to probabilities
# softmax_output = output.logits_labels.softmax(dim=1).cpu().numpy()

# for path, (p_real, p_fake) in zip(paths, softmax_output):
#     print(f"p(real) = {p_real:.4f}, p(fake) = {p_fake:.4f}, image: {path}")


# run inference on full validation set, with batch size 128, using dataloader to load images efficiently
# define a new dataloader class using the above preprocessing steps
class ImageDataset(Dataset):
    def __init__(self, img_path_df, preprocessing):
        self.img_path_df = img_path_df
        self.preprocessing = preprocessing
        self.img_dir_base = "/scratch/jlee436/cs584/data/"

    def __len__(self):
        return len(self.img_path_df)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir_base, self.img_path_df.iloc[idx]["file_name"]))
        label = self.img_path_df.iloc[idx]["label"]
        return self.preprocessing(image), label

val_ds = ImageDataset(df_val, preprocessing)

val_loader = DataLoader(val_ds, batch_size=128)
all_probs = []
all_labels = []

with torch.no_grad():
    for batch, labels in tqdm(val_loader):
        # Move batch to device and correct dtype
        batch = batch.to(fabric.device).to(model.dtype)
        
        # Forward pass
        output = model(batch)
        
        # Get probabilities
        probs = output.logits_labels.softmax(dim=1).cpu()
        
        all_probs.append(probs)
        all_labels.append(labels)

# Concatenate all predictions
all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)

# Save all_probs's first column and all_labeles into a csv file
df = pd.DataFrame({"probs": all_probs[:, 0], "labels": all_labels})
df.to_csv("all_probs_and_labels.csv", index=False)

# Print confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_probs.argmax(dim=1))
print(cm)


