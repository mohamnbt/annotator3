#!/usr/bin/env python3
"""
train_yolo.py  —  Script d'entraînement YOLOv8 segmentation
COSMER Annotator — Laboratoire COSMER, Université de Toulon

Usage:
    # Depuis la racine du ZIP extrait (où se trouve ce script) :
    python train_yolo.py
    python train_yolo.py --model yolov8s-seg.pt --epochs 200 --batch 8

Structure attendue :
    <dossier_extrait>/
    ├── dataset/
    │   ├── images/train/
    │   ├── images/val/
    │   ├── labels/train/
    │   ├── labels/val/
    │   ├── metadata/
    │   ├── dataset.yaml
    │   └── dataset_summary.csv
    └── train_yolo.py   ← ce script
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Entraîner YOLOv8 segmentation (câble sous-marin)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",   default="yolov8n-seg.pt",
                        help="Modèle de base : yolov8n/s/m/l/x-seg.pt")
    parser.add_argument("--epochs",  type=int, default=100,
                        help="Nombre d'epochs d'entraînement")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Taille des images (pixels)")
    parser.add_argument("--batch",   type=int, default=16,
                        help="Taille du batch (-1 = auto)")
    parser.add_argument("--device",  default="",
                        help="Device cible : 0 (GPU), cpu, mps (Apple Silicon)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Nb de workers DataLoader")
    parser.add_argument("--project", default="runs/segment",
                        help="Dossier racine des runs")
    parser.add_argument("--name",    default="cosmer_cable",
                        help="Nom de l'expérience")
    parser.add_argument("--resume",  action="store_true",
                        help="Reprendre un entraînement interrompu")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERREUR] ultralytics non installé.")
        print("         Installez-le avec : pip install ultralytics")
        return

    # Cherche dataset.yaml dans ./dataset/ (emplacement standard après extraction du ZIP)
    data_yaml = Path(__file__).parent / "dataset" / "dataset.yaml"
    if not data_yaml.exists():
        print(f"[ERREUR] dataset.yaml introuvable : {data_yaml}")
        print("         Vérifiez que le dossier dataset/ est au même niveau que ce script.")
        return

    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=True,
    )
    if args.device:
        train_kwargs["device"] = args.device

    print(f"\n🚀  Démarrage entraînement YOLOv8-seg")
    print(f"    Dataset  : {data_yaml}")
    print(f"    Modèle   : {args.model}")
    print(f"    Epochs   : {args.epochs}")
    print(f"    Img size : {args.imgsz}")
    print(f"    Batch    : {args.batch}")
    print(f"    Résultats: {args.project}/{args.name}\n")

    results = model.train(**train_kwargs)

    print("\n✅  Entraînement terminé.")
    print(f"    Meilleurs poids : {args.project}/{args.name}/weights/best.pt")

    # Validation finale
    print("\n📊  Validation finale...")
    metrics = model.val()
    print(f"    mAP50-95 seg : {metrics.seg.map:.4f}")
    print(f"    mAP50    seg : {metrics.seg.map50:.4f}")


if __name__ == "__main__":
    main()
