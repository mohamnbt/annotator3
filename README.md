# COSMER Annotator v2

> Outil d'annotation et d'entraînement IA pour l'estimation de vitesse de courant marin par analyse visuelle de câble de mouillage.  
> Laboratoire COSMER — Université de Toulon

## Installation

```bash
pip install -r requirements.txt
cd frontend && npm install
```

## Dépendances système (extraction vidéo)

```bash
brew install ffmpeg    # macOS
apt install ffmpeg     # Ubuntu/Debian
```

## Lancement

```bash
chmod +x start.sh && ./start.sh
# Ouvrir http://localhost:5173
```

## IA optionnelle (régression Vc + auto-annotation YOLO)

```bash
pip install torch torchvision          # entraînement + prédiction
pip install ultralytics                # auto-annotation YOLO
# Placer best.pt dans backend/best.pt
```

## Usage typique

1. Créer une session
2. Uploader des vidéos DJI → extraire les frames (ffmpeg)
3. Annoter les câbles + renseigner `current_speed_cm_s`
4. Exporter le dataset YOLO
5. Onglet **IA / Modèle** → Entraîner → Prédire sur nouvelle image

## Structure des données

```
data/
  sessions/{session_name}/
    images/          ← JPG/PNG
    labels/          ← .txt YOLO (auto-générés)
    annotations/     ← .json par image
    session.json
  models/            ← .pth entraînés
```
