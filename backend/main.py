"""
COSMER Annotator v2 — Backend FastAPI
Laboratoire COSMER, Université de Toulon
"""

import os
import re
import json
import csv
import io
import math
import shutil
import zipfile
import random
import unicodedata
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse

import cv2
import numpy as np

app = FastAPI(title="COSMER Annotator API v2", version="2.0.0")

# ─── Progression globals ────────────────────────────────────────────────────────
EXTRACTION_PROGRESS = {}  # {session_name: {current, total, status}}
TRAIN_PROGRESS = {}        # {session_name: {epoch, total_epochs, status, ...}}
GLOBAL_TRAIN_PROGRESS = {}  # {"global": {...}}

# ─── CORS ───────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Paths ──────────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── YOLO model (optional) ───────────────────────────────────────────────────────────────────────
YOLO_MODEL = None
YOLO_MODEL_PATH = Path(__file__).parent / "best.pt"

def get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None and YOLO_MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            YOLO_MODEL = YOLO(str(YOLO_MODEL_PATH))
        except Exception as e:
            print(f"[YOLO] Cannot load model: {e}")
    return YOLO_MODEL

try:
    get_yolo_model()
except Exception:
    pass


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ─── Helpers ─────────────────────────────────────────────────────────────────────────────────────

def ann_get(ann: dict, field: str, default=None):
    conditions = ann.get("conditions", {})
    if field in conditions:
        return conditions[field]
    return ann.get(field, default)


def sanitize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    ascii_str = ascii_str.replace(" ", "_")
    ascii_str = re.sub(r"[^a-zA-Z0-9_\-]", "", ascii_str)
    return ascii_str

def get_session_dir(name: str) -> Path:
    return DATA_DIR / name

def load_session_meta(name: str) -> dict:
    meta_path = get_session_dir(name) / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found")
    with open(meta_path, "r") as f:
        return json.load(f)

def save_session_meta(name: str, meta: dict):
    with open(get_session_dir(name) / "session.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

def write_yolo_label(session_name: str, stem: str, points: list, img_width: int, img_height: int):
    labels_dir = get_session_dir(session_name) / "labels"
    labels_dir.mkdir(exist_ok=True)
    label_path = labels_dir / f"{stem}.txt"
    if not points or len(points) < 2:
        return
    normalized = []
    for pt in points:
        x_norm = max(0.0, min(1.0, pt["x"] / img_width))
        y_norm = max(0.0, min(1.0, pt["y"] / img_height))
        normalized.append(f"{x_norm:.6f} {y_norm:.6f}")
    line = "0 " + " ".join(normalized) + "\n"
    with open(label_path, "w") as f:
        f.write(line)

def calc_cable_angle(points: list) -> dict:
    result = {"cable_angle_deg": "", "cable_angle_chord_deg": "", "cable_curvature_index": ""}
    if not points or len(points) < 2:
        return result
    xs = np.array([float(p["x"]) for p in points])
    ys = np.array([float(p["y"]) for p in points])
    coords = np.stack([xs, ys], axis=1)
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        cov = np.array([[float(cov), 0.0], [0.0, 0.0]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, np.argmax(eigenvalues)]
    dx_reg = float(principal[0])
    dy_reg = float(principal[1])
    if (ys[-1] - ys[0]) < 0:
        dy_reg, dx_reg = -dy_reg, -dx_reg
    angle_reg = math.degrees(math.atan2(abs(dx_reg), abs(dy_reg)))
    result["cable_angle_deg"] = round(angle_reg, 3)
    dx_chord = float(xs[-1] - xs[0])
    dy_chord = float(ys[0] - ys[-1])
    if abs(dx_chord) < 1e-6 and abs(dy_chord) < 1e-6:
        result["cable_angle_chord_deg"] = result["cable_angle_deg"]
    else:
        result["cable_angle_chord_deg"] = round(math.degrees(math.atan2(abs(dx_chord), abs(dy_chord))), 3)
    try:
        result["cable_curvature_index"] = round(abs(float(result["cable_angle_deg"]) - float(result["cable_angle_chord_deg"])), 3)
    except (TypeError, ValueError):
        result["cable_curvature_index"] = ""
    return result

def extract_centerline_from_mask(mask_xy: np.ndarray, img_w: int, img_h: int, n_points: int = 40) -> list:
    mask_img = np.zeros((img_h, img_w), dtype=np.uint8)
    pts = mask_xy.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_img, [pts], 255)
    try:
        from skimage.morphology import skeletonize
        skeleton = skeletonize(mask_img > 0).astype(np.uint8) * 255
    except ImportError:
        skeleton = mask_img.copy()
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return _centerline_slices(mask_img, img_w, img_h, n_points)
    skeleton_pts = np.stack([xs, ys], axis=1).astype(float)
    mean = skeleton_pts.mean(axis=0)
    centered = skeleton_pts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    projections = centered @ principal_axis
    order = np.argsort(projections)
    ordered_pts = skeleton_pts[order]
    if len(ordered_pts) > n_points:
        indices = np.linspace(0, len(ordered_pts) - 1, n_points, dtype=int)
        ordered_pts = ordered_pts[indices]
    return [{"x": float(p[0]), "y": float(p[1])} for p in ordered_pts]

def _centerline_slices(mask_img: np.ndarray, img_w: int, img_h: int, n_points: int) -> list:
    points = []
    ys, xs = np.where(mask_img > 0)
    if len(xs) == 0:
        return []
    span_x = xs.max() - xs.min()
    span_y = ys.max() - ys.min()
    if span_y >= span_x:
        y_min, y_max = int(ys.min()), int(ys.max())
        for i in range(n_points):
            y = int(y_min + (y_max - y_min) * i / (n_points - 1))
            row = np.where(mask_img[y, :] > 0)[0]
            if len(row) > 0:
                points.append({"x": float(row.mean()), "y": float(y)})
    else:
        x_min, x_max = int(xs.min()), int(xs.max())
        for i in range(n_points):
            x = int(x_min + (x_max - x_min) * i / (n_points - 1))
            col = np.where(mask_img[:, x] > 0)[0]
            if len(col) > 0:
                points.append({"x": float(x), "y": float(col.mean())})
    return points


# ─── Uniformisation des points ──────────────────────────────────────────────────────────────────

def _resample_equidistant(points: list, n: int) -> list:
    if len(points) < 2 or n < 2:
        return points
    coords = np.array([[p["x"], p["y"]] for p in points], dtype=float)
    diffs = np.diff(coords, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum[-1]
    if total < 1e-9:
        return points
    target = np.linspace(0.0, total, n)
    new_x = np.interp(target, cum, coords[:, 0])
    new_y = np.interp(target, cum, coords[:, 1])
    return [{"x": float(x), "y": float(y)} for x, y in zip(new_x, new_y)]


def _get_session_target_n(session_name: str, current_n: int, default_n: int = 40) -> int:
    ann_dir = get_session_dir(session_name) / "annotations"
    if not ann_dir.exists():
        return min(current_n, default_n)
    existing_counts = []
    for ann_file in ann_dir.glob("*.json"):
        try:
            with open(ann_file) as f:
                ann = json.load(f)
            pts = ann.get("points", [])
            if len(pts) >= 2:
                existing_counts.append(len(pts))
        except Exception:
            pass
    if not existing_counts:
        return min(current_n, default_n)
    target = min(min(existing_counts), current_n)
    return max(target, 2)


def _retroactively_resample_session(session_name: str, target_n: int):
    ann_dir = get_session_dir(session_name) / "annotations"
    if not ann_dir.exists():
        return
    for ann_file in ann_dir.glob("*.json"):
        try:
            with open(ann_file) as f:
                ann = json.load(f)
            pts = ann.get("points", [])
            if len(pts) == target_n or len(pts) < 2:
                continue
            new_pts = _resample_equidistant(pts, target_n)
            ann["points"] = new_pts
            ann["n_points_normalized"] = target_n
            angle_data = calc_cable_angle(new_pts)
            ann.update(angle_data)
            with open(ann_file, "w") as f:
                json.dump(ann, f, indent=2, ensure_ascii=False)
            stem = ann_file.stem
            img_w = ann.get("image_width", 1)
            img_h = ann.get("image_height", 1)
            write_yolo_label(session_name, stem, new_pts, img_w, img_h)
        except Exception as e:
            print(f"[RESAMPLE] Erreur sur {ann_file.name}: {e}")


# ─── PyTorch helper ──────────────────────────────────────────────────────────────────────────────────────
def get_torch_modules():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, models
        from PIL import Image
        return torch, nn, Dataset, DataLoader, transforms, models, Image
    except ImportError:
        return None, None, None, None, None, None, None


# ─── Helper : extraire les paires (theta, vc) ──────────────────────────────────────────────────
def _build_angle_vc_pts(samples: list) -> list:
    pts = []
    for img_path, vc in samples:
        p = Path(img_path)
        ann_path = p.parent.parent / "annotations" / (p.stem + ".json")
        if not ann_path.exists():
            continue
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            theta_raw = ann.get("cable_angle_deg")
            if theta_raw in ("", None):
                continue
            pts.append({"theta": round(float(theta_raw), 3), "vc": round(float(vc), 3)})
        except Exception:
            continue
    pts.sort(key=lambda d: d["theta"])
    return pts


# ─── Helper CSV pour export ─────────────────────────────────────────────────────────────────────
CSV_FIELDNAMES = [
    "filename", "session", "annotator_name", "current_speed_cm_s", "wave_amplitude_cm",
    "wave_length_cm", "wave_speed_cm_s", "current_direction", "camera_angle",
    "cable_angle_deg", "cable_angle_chord_deg", "cable_curvature_index",
    "water_depth_m", "cable_tension_n", "notes", "split",
]

def _build_csv(items: list) -> str:
    csv_buf = io.StringIO()
    writer = csv.DictWriter(csv_buf, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    for split_name, session_name, fn, ann_path in items:
        row = {"filename": fn, "session": session_name, "split": split_name}
        if ann_path.exists():
            with open(ann_path) as af:
                ann = json.load(af)
            flat = {k: v for k, v in ann.items() if k != "conditions"}
            row.update(flat)
            conditions = ann.get("conditions", {})
            for field in CSV_FIELDNAMES:
                if field not in row and field in conditions:
                    row[field] = conditions[field]
        writer.writerow(row)
    return csv_buf.getvalue()


# ─── Helper : génère le script train_yolo.py pour le ZIP ────────────────────────────────────────
def _build_train_script(dataset_name: str) -> str:
    return f'''#!/usr/bin/env python3
"""
train_yolo.py  —  Script d\'entraînement YOLOv8 segmentation
Dataset : {dataset_name}
Généré automatiquement par COSMER Annotator
Laboratoire COSMER, Université de Toulon
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Entraîner YOLOv8 seg sur le dataset {dataset_name}")
    parser.add_argument("--model",   default="yolov8n-seg.pt", help="Modèle de base (yolov8n/s/m/l/x-seg.pt)")
    parser.add_argument("--epochs",  type=int,   default=100,  help="Nombre d\'epochs")
    parser.add_argument("--imgsz",   type=int,   default=640,  help="Taille des images")
    parser.add_argument("--batch",   type=int,   default=16,   help="Taille du batch (-1 = auto)")
    parser.add_argument("--device",  default="",              help="Device : 0, cpu, mps ...")
    parser.add_argument("--workers", type=int,   default=4,    help="Nombre de workers DataLoader")
    parser.add_argument("--project", default="runs/segment",  help="Dossier de sortie")
    parser.add_argument("--name",    default="{dataset_name}", help="Nom de l\'expérience")
    parser.add_argument("--resume",  action="store_true",     help="Reprendre l\'entraînement")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERREUR] ultralytics non installé. Lancez : pip install ultralytics")
        return

    # train_yolo.py est DANS dataset/, donc dataset.yaml est au même niveau
    data_yaml = Path(__file__).parent / "dataset.yaml"
    if not data_yaml.exists():
        print(f"[ERREUR] dataset.yaml introuvable : {{data_yaml}}")
        print("Vérifiez que dataset.yaml est au même niveau que ce script (dans dataset/).")
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

    print(f"\\n🚀  Démarrage de l\'entraînement — dataset : {{data_yaml}}")
    print(f"    Modèle   : {{args.model}}")
    print(f"    Epochs   : {{args.epochs}}")
    print(f"    Img size : {{args.imgsz}}")
    print(f"    Batch    : {{args.batch}}")
    print(f"    Résultats: {{args.project}}/{{args.name}}\\n")

    results = model.train(**train_kwargs)
    print("\\n✅  Entraînement terminé.")
    print(f"    Meilleurs poids : {{args.project}}/{{args.name}}/weights/best.pt")

    # Validation finale
    metrics = model.val()
    print(f"\\n📊  Résultats validation :")
    print(f"    mAP50-95 seg : {{metrics.seg.map:.4f}}")
    print(f"    mAP50    seg : {{metrics.seg.map50:.4f}}")


if __name__ == "__main__":
    main()
'''


# ─── Session Routes ──────────────────────────────────────────────────────────────────────────────────────

@app.get("/api/sanitize")
async def sanitize_name_endpoint(name: str = Query(...)):
    return {"sanitized": sanitize_name(name)}

@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    if not DATA_DIR.exists():
        return sessions
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "session.json").exists():
            sessions.append(load_session_meta(d.name))
    return sessions

@app.post("/api/sessions")
async def create_session(name: str = Form(...), description: str = Form(""), folder: str = Form("")):
    sanitized = sanitize_name(name)
    if not sanitized:
        raise HTTPException(status_code=400, detail="Invalid session name")
    session_dir = get_session_dir(sanitized)
    if session_dir.exists():
        raise HTTPException(status_code=409, detail=f"Session '{sanitized}' already exists")
    session_dir.mkdir(parents=True)
    (session_dir / "images").mkdir()
    (session_dir / "labels").mkdir()
    (session_dir / "annotations").mkdir()
    meta = {
        "name": sanitized,
        "description": description,
        "folder": folder,
        "created_at": datetime.now().isoformat(),
        "images": [],
    }
    save_session_meta(sanitized, meta)
    return meta

@app.get("/api/sessions/{name}")
async def get_session(name: str):
    return load_session_meta(name)

@app.patch("/api/sessions/{name}")
async def update_session(name: str, folder: Optional[str] = Body(None), description: Optional[str] = Body(None)):
    meta = load_session_meta(name)
    if folder is not None:
        meta["folder"] = folder
    if description is not None:
        meta["description"] = description
    save_session_meta(name, meta)
    return meta

@app.delete("/api/sessions/{name}")
async def delete_session(name: str):
    session_dir = get_session_dir(name)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    shutil.rmtree(session_dir)
    return {"message": f"Session '{name}' deleted"}

@app.post("/api/sessions/batch-move")
async def batch_move_sessions(names: List[str] = Body(...), folder: str = Body(...)):
    updated = []
    for name in names:
        try:
            meta = load_session_meta(name)
            meta["folder"] = folder
            save_session_meta(name, meta)
            updated.append(name)
        except Exception:
            pass
    return {"updated": updated}


# ─── Image Routes ──────────────────────────────────────────────────────────────────────────────────────

@app.post("/api/sessions/{name}/images")
async def upload_images(name: str, files: List[UploadFile] = File(...)):
    meta = load_session_meta(name)
    images_dir = get_session_dir(name) / "images"
    uploaded = []
    for file in files:
        original = file.filename or "image.jpg"
        stem = Path(original).stem
        ext = Path(original).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)
        filename = f"{safe_stem}{ext}"
        counter = 1
        while filename in [img["filename"] for img in meta["images"]]:
            filename = f"{safe_stem}_{counter}{ext}"
            counter += 1
        filepath = images_dir / filename
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        meta["images"].append({"filename": filename, "status": "to_annotate", "added_at": datetime.now().isoformat()})
        uploaded.append(filename)
    save_session_meta(name, meta)
    return {"uploaded": uploaded, "count": len(uploaded)}

@app.get("/api/sessions/{name}/images/{filename}")
async def get_image(name: str, filename: str):
    filepath = get_session_dir(name) / "images" / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    ext = filepath.suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return StreamingResponse(open(filepath, "rb"), media_type=media_type)

@app.delete("/api/sessions/{name}/images/{filename}")
async def delete_image(name: str, filename: str):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    img_path = session_dir / "images" / filename
    if img_path.exists():
        os.remove(img_path)
    stem = Path(filename).stem
    for suffix, subdir in [(".json", "annotations"), (".txt", "labels")]:
        p = session_dir / subdir / f"{stem}{suffix}"
        if p.exists():
            os.remove(p)
    meta["images"] = [img for img in meta["images"] if img["filename"] != filename]
    save_session_meta(name, meta)
    return {"message": f"Image '{filename}' deleted"}


# ─── YOLO Auto-annotation ─────────────────────────────────────────────────────────────────────────

@app.get("/api/yolo/status")
async def yolo_status():
    return {
        "model_path": str(YOLO_MODEL_PATH),
        "model_exists": YOLO_MODEL_PATH.exists(),
        "model_loaded": YOLO_MODEL is not None,
    }

@app.get("/api/sessions/{name}/images/{filename}/predict")
async def predict_annotation(name: str, filename: str, conf: float = Query(0.5)):
    model = get_yolo_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle YOLO non disponible. Placez best.pt dans le dossier backend.")
    img_path = get_session_dir(name) / "images" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        results = model.predict(str(img_path), conf=conf, verbose=False)
        result = results[0]
        if result.masks is None or len(result.masks) == 0:
            return {"found": False, "points": [], "message": "Aucun objet détecté"}
        best_idx = int(result.boxes.conf.argmax())
        mask_xy = result.masks.xy[best_idx]
        img_w, img_h = result.orig_shape[1], result.orig_shape[0]
        points = extract_centerline_from_mask(mask_xy, img_w, img_h, n_points=40)
        if not points:
            points = [{"x": float(pt[0]), "y": float(pt[1])} for pt in mask_xy]
        conf_val = float(result.boxes.conf[best_idx])
        return {"found": True, "points": points, "image_width": img_w, "image_height": img_h, "confidence": conf_val, "message": f"{len(points)} points (centerline, conf={conf_val:.2f})"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur YOLO: {str(e)}")


# ─── Video Extraction (FFmpeg) ────────────────────────────────────────────────────────────────────────────────────

def extract_frames_ffmpeg_bg(video_path: str, output_dir: str, video_stem: str,
                              frame_interval: int, session_name: str):
    output_pattern = os.path.join(output_dir, f"{video_stem}_frame_%06d.jpg")
    try:
        probe = subprocess.run([
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-print_format", "default=noprint_wrappers=1:nokey=1", video_path
        ], capture_output=True, text=True, timeout=30)
        total_frames = int(probe.stdout.strip())
        expected = max(1, total_frames // frame_interval)
    except Exception:
        expected = 0

    EXTRACTION_PROGRESS[session_name] = {"current": 0, "total": expected, "status": "running"}

    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"select='not(mod(n\\,{frame_interval}))'",
        "-vsync", "vfr",
        "-q:v", "2",
        "-y", output_pattern
    ], capture_output=True, text=True, timeout=600)

    try:
        meta = load_session_meta(session_name)
        created_files = sorted(Path(output_dir).glob(f"{video_stem}_frame_*.jpg"))
        existing = {img["filename"] for img in meta["images"]}
        for f in created_files:
            if f.name not in existing:
                meta["images"].append({
                    "filename": f.name,
                    "status": "to_annotate",
                    "added_at": datetime.now().isoformat(),
                    "source_video": Path(video_path).name.replace("temp_", ""),
                })
        save_session_meta(session_name, meta)
        EXTRACTION_PROGRESS[session_name] = {"current": len(created_files), "total": len(created_files), "status": "done"}
    except Exception as e:
        EXTRACTION_PROGRESS[session_name] = {"status": "error", "error": str(e)}
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.get("/api/sessions/{name}/extraction-progress")
@app.get("/api/sessions/{name}/video/progress")
async def get_extraction_progress(name: str):
    return EXTRACTION_PROGRESS.get(name, {"current": 0, "total": 0, "status": "idle"})

@app.post("/api/sessions/{name}/video")
async def extract_video_frames(
    name: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    frame_interval: int = Form(240)
):
    s_dir = get_session_dir(name)
    images_dir = s_dir / "images"
    images_dir.mkdir(exist_ok=True)
    temp_video = s_dir / f"temp_{file.filename}"
    with open(temp_video, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    background_tasks.add_task(
        extract_frames_ffmpeg_bg,
        str(temp_video), str(images_dir),
        Path(file.filename).stem, frame_interval, name
    )
    return {"status": "started", "message": "Extraction lancée via FFmpeg en arrière-plan"}


# ─── Annotation Routes ────────────────────────────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/annotations/{stem}")
async def get_annotation(name: str, stem: str):
    ann_path = get_session_dir(name) / "annotations" / f"{stem}.json"
    if not ann_path.exists():
        return None
    with open(ann_path, "r") as f:
        return json.load(f)

@app.post("/api/sessions/{name}/annotations/{stem}")
async def save_annotation(name: str, stem: str, data: dict):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    ann_dir = session_dir / "annotations"
    ann_dir.mkdir(exist_ok=True)
    ann_path = ann_dir / f"{stem}.json"

    points = data.get("points", [])
    img_width = data.get("image_width", 1)
    img_height = data.get("image_height", 1)

    if points and len(points) >= 2:
        target_n = _get_session_target_n(name, len(points))
        points = _resample_equidistant(points, target_n)
        data["points"] = points
        data["n_points_normalized"] = target_n
        _retroactively_resample_session(name, target_n)

    if points and len(points) >= 2:
        angle_data = calc_cable_angle(points)
        data["cable_angle_deg"] = angle_data["cable_angle_deg"]
        data["cable_angle_chord_deg"] = angle_data["cable_angle_chord_deg"]
        data["cable_curvature_index"] = angle_data["cable_curvature_index"]

    data["saved_at"] = datetime.now().isoformat()

    with open(ann_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    mode = data.get("annotation_mode", "centerline")
    if mode == "contour" and "left_points" in data and "right_points" in data:
        left = data["left_points"]
        right = list(reversed(data["right_points"]))
        write_yolo_label(name, stem, left + right, img_width, img_height)
    else:
        write_yolo_label(name, stem, points, img_width, img_height)

    for img in meta["images"]:
        if Path(img["filename"]).stem == stem:
            img["status"] = "annotated"
            break
    save_session_meta(name, meta)
    return data

@app.post("/api/sessions/{name}/images/{filename}/ignore")
async def ignore_image(name: str, filename: str):
    meta = load_session_meta(name)
    for img in meta["images"]:
        if img["filename"] == filename:
            img["status"] = "ignored"
            break
    save_session_meta(name, meta)
    return {"message": f"Image '{filename}' ignored"}

@app.get("/api/sessions/{name}/last-conditions")
async def get_last_conditions(name: str):
    meta = load_session_meta(name)
    ann_dir = get_session_dir(name) / "annotations"
    last = {}
    for img in reversed(meta["images"]):
        if img["status"] == "annotated":
            stem = Path(img["filename"]).stem
            ann_path = ann_dir / f"{stem}.json"
            if ann_path.exists():
                with open(ann_path) as f:
                    ann = json.load(f)
                last = {
                    "current_direction": ann_get(ann, "current_direction", ""),
                    "camera_angle": ann_get(ann, "camera_angle", ""),
                    "annotator_name": ann_get(ann, "annotator_name", ""),
                }
                break
    return last


# ─── Export Routes ──────────────────────────────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/export/stats")
async def export_stats(name: str):
    meta = load_session_meta(name)
    annotated = [img for img in meta["images"] if img["status"] == "annotated"]
    total = len(annotated)
    train_count = int(total * 0.8)
    val_count = total - train_count
    return {
        "total_annotated": total,
        "total_images": len(meta["images"]),
        "train_count": train_count,
        "val_count": val_count,
    }

@app.get("/api/sessions/{name}/export/download")
async def export_download(name: str):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    annotated = [img for img in meta["images"] if img["status"] == "annotated"]
    if not annotated:
        raise HTTPException(status_code=404, detail="No annotated images")
    buf = io.BytesIO()
    random.shuffle(annotated)
    split = int(len(annotated) * 0.8)
    train_imgs = annotated[:split]
    val_imgs = annotated[split:]

    dataset_name = name
    ROOT = "dataset/"

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_items = []
        for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
            for img_meta in imgs:
                fn = img_meta["filename"]
                stem = Path(fn).stem
                img_path = session_dir / "images" / fn
                lbl_path = session_dir / "labels" / f"{stem}.txt"
                ann_path = session_dir / "annotations" / f"{stem}.json"
                if img_path.exists():
                    zf.write(img_path, f"{ROOT}images/{split_name}/{fn}")
                if lbl_path.exists():
                    zf.write(lbl_path, f"{ROOT}labels/{split_name}/{stem}.txt")
                if ann_path.exists():
                    zf.write(ann_path, f"{ROOT}metadata/{split_name}/{stem}.json")
                csv_items.append((split_name, name, fn, ann_path))

        yaml_content = (
            f"# Dataset : {dataset_name}\n"
            f"path: ./dataset\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 1\n"
            f"names: ['cable']\n"
        )
        zf.writestr(f"{ROOT}dataset.yaml", yaml_content)
        zf.writestr(f"{ROOT}dataset_summary.csv", _build_csv(csv_items))
        zf.writestr(f"{ROOT}train_yolo.py", _build_train_script(dataset_name))

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={name}_dataset.zip"},
    )

@app.get("/api/export/global/download")
async def global_export_download(sessions: Optional[List[str]] = Query(None)):
    if sessions:
        session_names = sessions
    else:
        session_names = [
            d.name for d in sorted(DATA_DIR.iterdir())
            if d.is_dir() and (d / "session.json").exists()
        ]

    all_annotated = []
    for sname in session_names:
        try:
            s = load_session_meta(sname)
        except Exception:
            continue
        for img in s["images"]:
            if img["status"] == "annotated":
                all_annotated.append((sname, img))

    if not all_annotated:
        raise HTTPException(status_code=404, detail="No annotated images across sessions")

    buf = io.BytesIO()
    random.shuffle(all_annotated)
    split = int(len(all_annotated) * 0.8)
    train_imgs = all_annotated[:split]
    val_imgs = all_annotated[split:]

    label = "_".join(session_names[:3]) if sessions else "global"
    dataset_name = label
    ROOT = "dataset/"

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_items = []
        for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
            for session_name, img_meta in imgs:
                session_dir = get_session_dir(session_name)
                fn = img_meta["filename"]
                stem = Path(fn).stem
                unique_fn = f"{session_name}__{fn}"
                unique_stem = f"{session_name}__{stem}"
                img_path = session_dir / "images" / fn
                lbl_path = session_dir / "labels" / f"{stem}.txt"
                ann_path = session_dir / "annotations" / f"{stem}.json"
                if img_path.exists():
                    zf.write(img_path, f"{ROOT}images/{split_name}/{unique_fn}")
                if lbl_path.exists():
                    zf.write(lbl_path, f"{ROOT}labels/{split_name}/{unique_stem}.txt")
                if ann_path.exists():
                    zf.write(ann_path, f"{ROOT}metadata/{split_name}/{unique_stem}.json")
                csv_items.append((split_name, session_name, unique_fn, ann_path))

        yaml_content = (
            f"# Dataset : {dataset_name}\n"
            f"path: ./dataset\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 1\n"
            f"names: ['cable']\n"
        )
        zf.writestr(f"{ROOT}dataset.yaml", yaml_content)
        zf.writestr(f"{ROOT}dataset_summary.csv", _build_csv(csv_items))
        zf.writestr(f"{ROOT}train_yolo.py", _build_train_script(dataset_name))

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={label}_dataset.zip"},
    )


# ─── Statistics ──────────────────────────────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/statistics")
async def get_statistics(name: str):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    ann_dir = session_dir / "annotations"
    total = len(meta["images"])
    annotated_imgs = [img for img in meta["images"] if img["status"] == "annotated"]
    ignored_imgs = [img for img in meta["images"] if img["status"] == "ignored"]
    annotations = []
    for img in annotated_imgs:
        stem = Path(img["filename"]).stem
        ann_path = ann_dir / f"{stem}.json"
        if ann_path.exists():
            with open(ann_path) as f:
                annotations.append(json.load(f))
    def histo(values, edges):
        result = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            label = f"{lo}-{hi}"
            count = sum(1 for v in values if lo <= v < hi)
            result.append({"range": label, "count": count})
        return result
    def count_field(field):
        from collections import Counter
        vals = [ann_get(a, field, "") for a in annotations if ann_get(a, field, "")]
        return [{"name": k, "value": v} for k, v in Counter(vals).most_common()]
    def safe_float(a, field):
        v = ann_get(a, field)
        if v in ("", None):
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
    speeds = [v for a in annotations if (v := safe_float(a, "current_speed_cm_s")) is not None]
    curvatures = [v for a in annotations if (v := safe_float(a, "cable_curvature_index")) is not None]
    angles = [v for a in annotations if (v := safe_float(a, "cable_angle_deg")) is not None]
    avg_pts = sum(len(a.get("points", [])) for a in annotations) / max(len(annotations), 1)
    wave_scatter = []
    for a in annotations:
        amp = safe_float(a, "wave_amplitude_cm")
        spd = safe_float(a, "current_speed_cm_s")
        if amp is not None and spd is not None:
            wave_scatter.append({"amplitude": amp, "speed": spd})
    balance_warnings = []
    if speeds:
        from collections import Counter
        speed_counts = Counter([round(s) for s in speeds])
        max_c = max(speed_counts.values())
        min_c = min(speed_counts.values())
        if max_c > min_c * 3:
            balance_warnings.append("Déséquilibre important dans la distribution des vitesses.")
    return {
        "total": total,
        "annotated": len(annotated_imgs),
        "ignored": len(ignored_imgs),
        "remaining": total - len(annotated_imgs) - len(ignored_imgs),
        "speed_histogram": histo(speeds, [0, 5, 10, 20, 30, 50, 100]),
        "curvature_histogram": histo(curvatures, [0, 1, 2, 5, 10, 20, 50]),
        "angle_histogram": histo(angles, [0, 5, 10, 15, 20, 30, 45, 90]),
        "camera_angles": count_field("camera_angle"),
        "current_directions": count_field("current_direction"),
        "wave_scatter": wave_scatter,
        "avg_points": round(avg_pts, 1),
        "annotators": count_field("annotator_name"),
        "balance_warnings": balance_warnings,
    }


# ─── Train Vc (ResNet18 per session) ────────────────────────────────────────────────────────────

def run_train_session(session_name: str, epochs: int):
    torch, nn, Dataset, DataLoader, transforms, models, Image = get_torch_modules()
    if torch is None:
        TRAIN_PROGRESS[session_name] = {"epoch": 0, "total_epochs": epochs, "status": "error", "error": "PyTorch non installé"}
        return
    try:
        meta = load_session_meta(session_name)
        session_dir = get_session_dir(session_name)
        ann_dir = session_dir / "annotations"
        images_dir = session_dir / "images"
        samples = []
        for img in meta["images"]:
            if img["status"] != "annotated":
                continue
            stem = Path(img["filename"]).stem
            ann_path = ann_dir / f"{stem}.json"
            img_path = images_dir / img["filename"]
            if not ann_path.exists() or not img_path.exists():
                continue
            with open(ann_path) as f:
                ann = json.load(f)
            vc_raw = ann_get(ann, "current_speed_cm_s")
            try:
                vc = float(vc_raw)
            except (TypeError, ValueError):
                continue
            samples.append((str(img_path), vc))
        if len(samples) < 4:
            TRAIN_PROGRESS[session_name] = {"epoch": 0, "total_epochs": epochs, "status": "error",
                                             "error": f"Pas assez d'images annotées avec Vc ({len(samples)} trouvées, minimum 4)"}
            return
        random.shuffle(samples)
        split = max(1, int(len(samples) * 0.8))
        train_samples = samples[:split]
        val_samples = samples[split:] if len(samples) > split else samples[-1:]
        all_angle_vc_pts = _build_angle_vc_pts(samples)
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
        class VcDataset(Dataset):
            def __init__(self, s, tf):
                self.s = s
                self.tf = tf
            def __len__(self):
                return len(self.s)
            def __getitem__(self, i):
                path, vc = self.s[i]
                img = Image.open(path).convert("RGB")
                return self.tf(img), torch.tensor([vc], dtype=torch.float32)
        train_loader = DataLoader(VcDataset(train_samples, tf_train), batch_size=min(8, len(train_samples)), shuffle=True)
        val_loader = DataLoader(VcDataset(val_samples, tf_val), batch_size=min(8, len(val_samples)))
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
        criterion = nn.MSELoss()
        train_losses, val_losses = [], []
        TRAIN_PROGRESS[session_name] = {
            "epoch": 0, "total_epochs": epochs, "status": "running",
            "n_train": len(train_samples), "n_val": len(val_samples),
            "device": str(device),
            "angle_vc_pts": all_angle_vc_pts,
        }
        for ep in range(epochs):
            model.train()
            tl = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                tl += loss.item()
            scheduler.step()
            model.eval()
            vl, preds_list, true_list = 0.0, [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    vl += criterion(out, yb).item()
                    preds_list += out.squeeze().cpu().tolist() if out.squeeze().dim() > 0 else [out.squeeze().item()]
                    true_list += yb.squeeze().cpu().tolist() if yb.squeeze().dim() > 0 else [yb.squeeze().item()]
            train_losses.append(round(tl / len(train_loader), 4))
            val_losses.append(round(vl / len(val_loader), 4))
            TRAIN_PROGRESS[session_name].update({"epoch": ep + 1, "train_losses": train_losses, "val_losses": val_losses})
        preds_arr = np.array(preds_list)
        true_arr = np.array(true_list)
        mae = float(np.mean(np.abs(preds_arr - true_arr)))
        rmse = float(np.sqrt(np.mean((preds_arr - true_arr) ** 2)))
        model_filename = f"{session_name}_vc_model.pth"
        torch.save(model.state_dict(), MODELS_DIR / model_filename)
        all_loader = DataLoader(VcDataset(samples, tf_val), batch_size=min(16, len(samples)))
        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, _ in all_loader:
                out = model(xb.to(device))
                all_preds += out.squeeze().cpu().tolist() if out.squeeze().dim() > 0 else [out.squeeze().item()]
        nn_angle_vc_pts = []
        for i, (img_path, _) in enumerate(samples):
            p = Path(img_path)
            ann_path = p.parent.parent / "annotations" / (p.stem + ".json")
            if not ann_path.exists():
                continue
            try:
                with open(ann_path) as f:
                    ann = json.load(f)
                theta_raw = ann.get("cable_angle_deg")
                if theta_raw in ("", None):
                    continue
                nn_angle_vc_pts.append({"theta": round(float(theta_raw), 3), "vc": round(float(all_preds[i]), 3)})
            except Exception:
                continue
        nn_angle_vc_pts.sort(key=lambda d: d["theta"])
        TRAIN_PROGRESS[session_name].update({
            "status": "done", "mae": round(mae, 3), "rmse": round(rmse, 3),
            "preds": [round(float(p), 2) for p in preds_list],
            "true": [round(float(t), 2) for t in true_list],
            "model_name": session_name,
            "angle_vc_pts": all_angle_vc_pts,
            "nn_angle_vc_pts": nn_angle_vc_pts,
        })
    except Exception as e:
        TRAIN_PROGRESS[session_name] = {"epoch": 0, "total_epochs": epochs, "status": "error", "error": str(e)}

@app.post("/api/sessions/{name}/train")
async def train_session_model(name: str, background_tasks: BackgroundTasks, epochs: int = Form(50)):
    load_session_meta(name)
    TRAIN_PROGRESS[name] = {"epoch": 0, "total_epochs": epochs, "status": "starting"}
    background_tasks.add_task(run_train_session, name, epochs)
    return {"status": "started", "session": name, "epochs": epochs}

@app.get("/api/sessions/{name}/train/progress")
async def get_train_progress(name: str):
    return TRAIN_PROGRESS.get(name, {"epoch": 0, "total_epochs": 0, "status": "idle"})


# ─── Angle-Vc data (par session) ─────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/angle-vc-data")
async def get_session_angle_vc_data(name: str):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    ann_dir = session_dir / "annotations"
    result = []
    for img in meta["images"]:
        if img["status"] != "annotated":
            continue
        stem = Path(img["filename"]).stem
        ann_path = ann_dir / f"{stem}.json"
        if not ann_path.exists():
            continue
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            theta_raw = ann.get("cable_angle_deg")
            vc_raw = ann_get(ann, "current_speed_cm_s")
            if theta_raw in ("", None) or vc_raw in ("", None):
                continue
            result.append({"theta": round(float(theta_raw), 3), "vc": round(float(vc_raw), 3)})
        except (ValueError, TypeError, KeyError):
            continue
    result.sort(key=lambda d: d["theta"])
    return result


# ─── Train Global Vc ──────────────────────────────────────────────────────────────────────────────

def run_train_global(session_names: Optional[List[str]], epochs: int, model_name: str):
    torch, nn, Dataset, DataLoader, transforms, models, Image = get_torch_modules()
    key = "global"
    if torch is None:
        GLOBAL_TRAIN_PROGRESS[key] = {"epoch": 0, "total_epochs": epochs, "status": "error", "error": "PyTorch non installé"}
        return
    try:
        if session_names:
            sessions_to_use = session_names
        else:
            sessions_to_use = [d.name for d in sorted(DATA_DIR.iterdir()) if d.is_dir() and (d / "session.json").exists()]
        samples = []
        for sname in sessions_to_use:
            try:
                meta = load_session_meta(sname)
            except Exception:
                continue
            session_dir = get_session_dir(sname)
            ann_dir = session_dir / "annotations"
            images_dir = session_dir / "images"
            for img in meta["images"]:
                if img["status"] != "annotated":
                    continue
                stem = Path(img["filename"]).stem
                ann_path = ann_dir / f"{stem}.json"
                img_path = images_dir / img["filename"]
                if not ann_path.exists() or not img_path.exists():
                    continue
                with open(ann_path) as f:
                    ann = json.load(f)
                vc_raw = ann_get(ann, "current_speed_cm_s")
                try:
                    vc = float(vc_raw)
                except (TypeError, ValueError):
                    continue
                samples.append((str(img_path), vc))
        if len(samples) < 4:
            GLOBAL_TRAIN_PROGRESS[key] = {"epoch": 0, "total_epochs": epochs, "status": "error",
                                           "error": f"Pas assez d'images ({len(samples)} trouvées, minimum 4)"}
            return
        random.shuffle(samples)
        split = max(1, int(len(samples) * 0.8))
        train_samples = samples[:split]
        val_samples = samples[split:] if len(samples) > split else samples[-1:]
        all_angle_vc_pts = _build_angle_vc_pts(samples)
        tf_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tf_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        class VcDataset(Dataset):
            def __init__(self, s, tf):
                self.s = s
                self.tf = tf
            def __len__(self):
                return len(self.s)
            def __getitem__(self, i):
                path, vc = self.s[i]
                img = Image.open(path).convert("RGB")
                return self.tf(img), torch.tensor([vc], dtype=torch.float32)
        train_loader = DataLoader(VcDataset(train_samples, tf_train), batch_size=min(16, len(train_samples)), shuffle=True)
        val_loader = DataLoader(VcDataset(val_samples, tf_val), batch_size=min(16, len(val_samples)))
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
        criterion = nn.MSELoss()
        train_losses, val_losses = [], []
        GLOBAL_TRAIN_PROGRESS[key] = {
            "epoch": 0, "total_epochs": epochs, "status": "running",
            "n_train": len(train_samples), "n_val": len(val_samples),
            "device": str(device),
            "angle_vc_pts": all_angle_vc_pts,
        }
        for ep in range(epochs):
            model.train()
            tl = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                tl += loss.item()
            scheduler.step()
            model.eval()
            vl, preds_list, true_list = 0.0, [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    vl += criterion(out, yb).item()
                    preds_list += out.squeeze().cpu().tolist() if out.squeeze().dim() > 0 else [out.squeeze().item()]
                    true_list += yb.squeeze().cpu().tolist() if yb.squeeze().dim() > 0 else [yb.squeeze().item()]
            train_losses.append(round(tl / len(train_loader), 4))
            val_losses.append(round(vl / len(val_loader), 4))
            GLOBAL_TRAIN_PROGRESS[key].update({"epoch": ep + 1, "train_losses": train_losses, "val_losses": val_losses})
        preds_arr = np.array(preds_list)
        true_arr = np.array(true_list)
        mae = float(np.mean(np.abs(preds_arr - true_arr)))
        rmse = float(np.sqrt(np.mean((preds_arr - true_arr) ** 2)))
        model_filename = f"{model_name}.pth"
        torch.save(model.state_dict(), MODELS_DIR / model_filename)
        all_loader = DataLoader(VcDataset(samples, tf_val), batch_size=min(16, len(samples)))
        model.eval()
        all_preds = []
        with torch.no_grad():
            for xb, _ in all_loader:
                out = model(xb.to(device))
                all_preds += out.squeeze().cpu().tolist() if out.squeeze().dim() > 0 else [out.squeeze().item()]
        nn_angle_vc_pts = []
        for i, (img_path, _) in enumerate(samples):
            p = Path(img_path)
            ann_path = p.parent.parent / "annotations" / (p.stem + ".json")
            if not ann_path.exists():
                continue
            try:
                with open(ann_path) as f:
                    ann = json.load(f)
                theta_raw = ann.get("cable_angle_deg")
                if theta_raw in ("", None):
                    continue
                nn_angle_vc_pts.append({"theta": round(float(theta_raw), 3), "vc": round(float(all_preds[i]), 3)})
            except Exception:
                continue
        nn_angle_vc_pts.sort(key=lambda d: d["theta"])
        GLOBAL_TRAIN_PROGRESS[key].update({
            "status": "done", "mae": round(mae, 3), "rmse": round(rmse, 3),
            "preds": [round(float(p), 2) for p in preds_list],
            "true": [round(float(t), 2) for t in true_list],
            "model_name": model_name,
            "angle_vc_pts": all_angle_vc_pts,
            "nn_angle_vc_pts": nn_angle_vc_pts,
        })
    except Exception as e:
        GLOBAL_TRAIN_PROGRESS[key] = {"epoch": 0, "total_epochs": epochs, "status": "error", "error": str(e)}

@app.post("/api/train/global")
async def train_global(
    background_tasks: BackgroundTasks,
    sessions: Optional[str] = Form(None),
    epochs: int = Form(50),
    model_name: str = Form("global_vc_model"),
):
    session_list = [s.strip() for s in sessions.split(",") if s.strip()] if sessions else None
    GLOBAL_TRAIN_PROGRESS["global"] = {"epoch": 0, "total_epochs": epochs, "status": "starting"}
    background_tasks.add_task(run_train_global, session_list, epochs, model_name)
    return {"status": "started", "sessions": session_list, "epochs": epochs, "model_name": model_name}

@app.get("/api/train/global/progress")
async def get_global_train_progress():
    return GLOBAL_TRAIN_PROGRESS.get("global", {"epoch": 0, "total_epochs": 0, "status": "idle"})


@app.get("/api/train/global/angle-vc-data")
async def get_angle_vc_data():
    result = []
    if not DATA_DIR.exists():
        return result
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not (d / "session.json").exists():
            continue
        try:
            meta = load_session_meta(d.name)
        except Exception:
            continue
        ann_dir = d / "annotations"
        for img in meta["images"]:
            if img["status"] != "annotated":
                continue
            stem = Path(img["filename"]).stem
            ann_path = ann_dir / f"{stem}.json"
            if not ann_path.exists():
                continue
            try:
                with open(ann_path) as f:
                    ann = json.load(f)
                theta_raw = ann.get("cable_angle_deg")
                vc_raw = ann_get(ann, "current_speed_cm_s")
                if theta_raw in ("", None) or vc_raw in ("", None):
                    continue
                result.append({"theta": round(float(theta_raw), 3), "vc": round(float(vc_raw), 3)})
            except (ValueError, TypeError, KeyError):
                continue
    result.sort(key=lambda d: d["theta"])
    return result


# ─── Models ────────────────────────────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    models_list = []
    for f in sorted(MODELS_DIR.glob("*.pth")):
        stem = f.stem
        is_global = not stem.endswith("_vc_model") or stem.startswith("global")
        models_list.append({
            "name": stem,
            "filename": f.name,
            "size_mb": round(f.stat().st_size / 1024 / 1024, 2),
            "modified_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "is_global": is_global,
        })
    return models_list

@app.get("/api/models/{filename}/download")
async def download_model(filename: str):
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Modèle introuvable")
    return FileResponse(
        path=str(model_path),
        media_type="application/octet-stream",
        filename=filename,
    )


# ─── Predict Vc ──────────────────────────────────────────────────────────────────────────────────

@app.post("/api/predict")
async def predict_vc(model_name: str = Form(...), file: UploadFile = File(...)):
    torch, nn, Dataset, DataLoader, transforms, models, Image = get_torch_modules()
    if torch is None:
        raise HTTPException(status_code=503, detail="PyTorch non installé")
    model_path = MODELS_DIR / f"{model_name}.pth"
    if not model_path.exists():
        model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Modèle '{model_name}' introuvable")
    try:
        content = await file.read()
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(content)).convert("RGB")
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = tf(img).unsqueeze(0)
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        net = models.resnet18(weights=None)
        net.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        net.load_state_dict(torch.load(model_path, map_location=device))
        net = net.to(device)
        net.eval()
        with torch.no_grad():
            pred = net(x.to(device)).item()
        return {"vitesse_estimee": round(pred, 2), "model_used": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")
