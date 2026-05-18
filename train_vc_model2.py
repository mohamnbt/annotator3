# Script d'entraînement pour le modèle : vc_model2
# Généré par COSMER Annotator
#
# Ce script utilise un ResNet18 pour prédire la vitesse du courant (Vc)
# à partir des images de câbles sous-marins.
#
# INSTRUCTIONS :
# 1. Exportez vos données depuis l'application (Export Global ou Session).
# 2. Décompressez le zip dans un dossier 'dataset'.
# 3. Placez ce script à côté du dossier 'dataset'.
# 4. Installez les dépendances : pip install torch torchvision pandas Pillow
# 5. Lancez le script : python train_vc_model2.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import time

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
CSV_PATH = os.path.join(DATASET_PATH, "annotations.csv")
MODEL_SAVE_PATH = "vc_model2_retrained.pth"
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- Entraînement de vc_model2 ---")
print(f"Device : {DEVICE}")

class VcDataset(Dataset):
    def __init__(self, csv_data, transform=None):
        self.data = csv_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        split = row['split']
        filename = row['filename']
        img_path = os.path.join(DATASET_PATH, "images", split, filename)
        
        image = Image.open(img_path).convert("RGB")
        vc = torch.tensor([float(row['current_speed_cm_s'])], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, vc

# --- CHARGEMENT DES DONNÉES ---
if not os.path.exists(CSV_PATH):
    print(f"ERREUR : Fichier {CSV_PATH} introuvable. Avez-vous décompressé l'export dans le dossier '{DATASET_PATH}' ?")
    exit(1)

df = pd.read_csv(CSV_PATH)
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']

print(f"Train samples : {len(train_df)}")
print(f"Val samples   : {len(val_df)}")

tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tf_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_loader = DataLoader(VcDataset(train_df, tf_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(VcDataset(val_df, tf_val), batch_size=BATCH_SIZE)

# --- MODÈLE ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1)
)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# --- BOUCLE D'ENTRAÎNEMENT ---
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # Train
    model.train()
    train_loss = 0
    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    duration = time.time() - start_time
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f} ({duration:.1f}s)")
    
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  --> Modèle sauvegardé sous {MODEL_SAVE_PATH}")

print("\nEntraînement terminé !")
