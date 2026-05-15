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

# ─── Progression globals ───────────────────────────────────────────────────────
EXTRACTION_PROGRESS = {}  # {session_name: {current, total, status}}
TRAIN_PROGRESS = {}        # {session_name: {epoch, total_epochs, status, ...}}
GLOBAL_TRAIN_PROGRESS = {}  # {"global": {...}}

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── YOLO model (optional) ────────────────────────────────────────────────────
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

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


# ─── PyTorch helper ───────────────────────────────────────────────────────────
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


# ─── Session Routes ────────────────────────────────────────────────────────────

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


# ─── Image Routes ─────────────────────────────────────────────────────────────

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


# ─── YOLO Auto-annotation ──────────────────────────────────────────────────────

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


# ─── Video Extraction (FFmpeg) ─────────────────────────────────────────────────

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


# ─── Annotation Routes ─────────────────────────────────────────────────────────

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
    if points and len(points) >= 2:
        angle_data = calc_cable_angle(points)
        data["cable_angle_deg"] = angle_data["cable_angle_deg"]
        data["cable_angle_chord_deg"] = angle_data["cable_angle_chord_deg"]
        data["cable_curvature_index"] = angle_data["cable_curvature_index"]
    data["saved_at"] = datetime.now().isoformat()
    with open(ann_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    img_width = data.get("image_width", 1)
    img_height = data.get("image_height", 1)
    mode = data.get("annotation_mode", "centerline")
    if mode == "contour" and "left_points" in data and "right_points" in data:
        left = data["left_points"]
        right = list(reversed(data["right_points"]))
        write_yolo_label(name, stem, left + right, img_width, img_height)
    else:
        write_yolo_label(name, stem, points, img_width, img_height)
    for img in meta["images"]:
        if Path(img["filename"]).stem 