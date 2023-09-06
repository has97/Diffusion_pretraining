import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
import sys
import os
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/solo')

from solo.data.classification_dataloader import prepare_data

# change this if you wanna load a different model
my_backbone = "resnet18"

backbone_model = {
    "resnet18": resnet18,
    "resnet50": resnet50,
}[my_backbone]

# initialize backbone
kwargs = {
    "cifar": False,  # <-- change this if you are running on cifar
    # "img_size": 224,  # <-- uncomment this when using vit/swin
    # "patch_size": 16,  # <-- uncomment this when using vit
}
cifar = kwargs.pop("cifar", False)
# swin specific
if "swin" in my_backbone and cifar:
    kwargs["window_size"] = 4

model = backbone_model(**kwargs)
if "resnet" in my_backbone:
    # remove fc layer
    model.fc = nn.Identity()
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()
model2 = backbone_model(**kwargs)
if "resnet" in my_backbone:
    # remove fc layer
    model2.fc = nn.Identity()
    if cifar:
        model2.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model2.maxpool = nn.Identity()
ckpt_path ="/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/trained_models/byol/dcaesu55/byol-imagenet100_diffused_actual_more_domain-dcaesu55-ep=399.ckpt"
ckpt_path_1 = "/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/pretrained_checkpts/byol/byol-400ep-imagenet100-17qno0qt-ep=399.ckpt"

state = torch.load(ckpt_path)["state_dict"]
for k in list(state.keys()):
    if "backbone" in k:
        state[k.replace("backbone.", "")] = state[k]
    del state[k]
model.load_state_dict(state, strict=False)

state = torch.load(ckpt_path_1)["state_dict"]
for k in list(state.keys()):
    if "backbone" in k:
        state[k.replace("backbone.", "")] = state[k]
    del state[k]
model2.load_state_dict(state, strict=False)

print(f"loaded {ckpt_path}")
train_loader , val_loader = prepare_data(
    "custom",
    train_data_path="/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/domainnet/domainnet-quickdraw/train",
    val_data_path="/raid/biplab/phduser1/Hassan/diffused-solo-learn-main_1/diffused-solo-learn-main/domainnet/domainnet-quickdraw/val",
    batch_size=64,
    num_workers=4,
    auto_augment=True
)
model.eval()  # Set the model to evaluation mode
model=model.to('cuda:1')
model2.eval()  # Set the model to evaluation mode
model2=model2.to('cuda:1')
embeddings = []
embeddings1 = []
labels=[]
with torch.no_grad():
    for images, label in tqdm(val_loader):
        images= images.to('cuda:1')
        outputs = model(images)
        outputs1 = model2(images)
        labels.extend(label.numpy())
        embeddings.extend(outputs.cpu().numpy())  # Assuming 'outputs' is the embedding tensor
        embeddings1.extend(outputs1.cpu().numpy())

# Step 3: Perform t-SNE
unique_classes = np.unique(labels)
high_contrast_colors = [
    (1.0, 0.0, 0.0),    # Bright Red
    (0.0, 0.0, 1.0),    # Vibrant Blue
    (0.0, 1.0, 0.0),    # Lime Green
    (0.5, 0.0, 0.5),    # Deep Purple
    (1.0, 0.84, 0.0),   # Golden Yellow
    (0.0, 1.0, 1.0),    # Cyan
    (1.0, 0.0, 1.0),    # Magenta
    (0.0, 0.79, 0.34),  # Emerald Green
    (1.0, 0.65, 0.0),   # Orange
    (1.0, 0.41, 0.71)   # Hot Pink
]
# Randomly select 10 classes
selected_classes = np.random.choice(unique_classes, size=10, replace=False)
unique_classes = np.unique(labels)
t=0
cs={}
# print(len( unique_classes))
for u in selected_classes:
    cs[u]=high_contrast_colors[t]
    t+=1
# selected_classes = [class_label_1, class_label_2, ...]
mask = np.isin(labels, selected_classes)
# print(mask)np.where(c, a, b)
embeddings_s = [val for i, val in enumerate(embeddings) if mask[i]]
embeddings1_s = [val for i, val in enumerate(embeddings1) if mask[i]]
# embeddings_s = embeddings[mask]
labels_s = [cs[val] for i, val in enumerate(labels) if mask[i]]

# labels = labels[mask]
tsne = TSNE(n_components=2, random_state=42)
embedded_data = tsne.fit_transform(embeddings_s)
embedded1_data = tsne.fit_transform(embeddings1_s)
# print(embedded_data)
# Step 4: Create Plot
plt.figure(figsize=(8, 6))
plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels_s, marker='o')
plt.title("t-SNE Plot of Model Embeddings of subset of DomainNet-quickdraw trained with Diffused Images")
plt.savefig("./tsne_plot_3.eps")
plt.close()
plt.figure(figsize=(8, 6))
plt.scatter(embedded1_data[:, 0], embedded1_data[:, 1], c=labels_s, cmap='viridis', marker='o')
plt.title("t-SNE Plot of Model Embeddings of subset of DomaiNet-quickdraw trained with Imagenet-100")
plt.savefig("./tsne_plot_4.eps")
plt.close()
# # Close the plot to release resources
# plt.close()
# from solo.utils.auto_umap import OfflineUMAP

# umap = OfflineUMAP()

# # move model to the gpu
# device = "cuda:1"
# model = model.to(device)

# umap.plot(device, model, train_loader, 'im100_no_train_umap.pdf')
# umap.plot(device, model, val_loader, 'im100_no_val_umap.pdf')