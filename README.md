# 🤿 COSMER Annotator

**Outil d'annotation de lignes d'amarrage** pour le Laboratoire COSMER, Université de Toulon.

Crée un dataset annoté pour l'entraînement de modèles YOLOv8 segmentation.

## Installation

### Backend (Python)

```bash
cd backend
pip install -r requirements.txt
```

### Frontend (React + TypeScript)

```bash
cd frontend
npm install
```

## Lancement

```bash
chmod +x start.sh
./start.sh
```

Ouvrir **http://localhost:5173** dans le navigateur.

## Architecture

```
data/sessions/{session_name}/
  ├── images/          # Images JPG/PNG
  ├── labels/          # Fichiers .txt YOLO (générés automatiquement)
  ├── annotations/     # JSON par image
  └── session.json     # Métadonnées de session
```

## Export

Le dataset exporté est au format YOLOv8 segmentation :

```
dataset/
  ├── images/train/
  ├── images/val/
  ├── labels/train/
  ├── labels/val/
  ├── metadata/
  ├── dataset.yaml
  ├── dataset_summary.csv
  └── train_yolo.py
```

## Raccourcis clavier

| Touche | Action |
|--------|--------|
| Z | Annuler dernier point |
| Escape | Effacer tous les points |
| Enter | Confirmer annotation |
| ← → | Image précédente / suivante |
| Molette | Zoom |
| Espace + glisser | Panoramique |
| ? | Afficher les raccourcis |
# annotator3
