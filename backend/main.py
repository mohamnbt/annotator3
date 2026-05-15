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
from fastapi.responses import StreamingResponse, JSONResponse

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
        if Path(img["filename"]).stem == stem:
            img["status"] = "annotated"
            break
    save_session_meta(name, meta)
    return {"message": "Annotation saved", "stem": stem}

@app.post("/api/sessions/{name}/images/{filename}/ignore")
async def ignore_image(name: str, filename: str):
    meta = load_session_meta(name)
    for img in meta["images"]:
        if img["filename"] == filename:
            img["status"] = "ignored"
            break
    else:
        raise HTTPException(status_code=404, detail="Image not found")
    save_session_meta(name, meta)
    return {"message": f"Image '{filename}' marked as ignored"}

@app.get("/api/sessions/{name}/last-conditions")
async def get_last_conditions(name: str):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    annotated = [img for img in meta["images"] if img.get("status") == "annotated"]
    if not annotated:
        return None
    latest = None
    latest_time = ""
    for img in annotated:
        stem = Path(img["filename"]).stem
        ann_path = session_dir / "annotations" / f"{stem}.json"
        if ann_path.exists():
            with open(ann_path, "r") as f:
                ann = json.load(f)
            saved_at = ann.get("saved_at", "")
            if saved_at > latest_time:
                latest_time = saved_at
                latest = ann
    if latest:
        conditions = latest.get("conditions", {})
        if not conditions:
            # Try flat fields
            flat_keys = ["annotator_name", "current_speed_cm_s", "current_direction",
                         "wave_amplitude_cm", "wave_frequency_hz", "wind_speed_m_s",
                         "camera_angle", "water_turbidity", "lighting_condition",
                         "immersed_length_cm", "buoy_to_surface_cm", "canal_water_depth_cm", "notes"]
            conditions = {k: latest.get(k, "") for k in flat_keys if k in latest}
        return conditions
    return None


# ─── Statistics ───────────────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/statistics")
@app.get("/api/sessions/{name}/stats")
async def get_statistics(name: str):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    total = len(meta["images"])
    annotated_list = [img for img in meta["images"] if img.get("status") == "annotated"]
    annotated_count = len(annotated_list)
    ignored_count = sum(1 for img in meta["images"] if img.get("status") == "ignored")

    speeds, camera_angles, directions, wave_scatter, point_counts, annotators, curvatures, angles = [], {}, {}, [], [], {}, [], []

    for img in annotated_list:
        stem = Path(img["filename"]).stem
        ann_path = session_dir / "annotations" / f"{stem}.json"
        if ann_path.exists():
            with open(ann_path) as f:
                ann = json.load(f)
            cond = ann.get("conditions", {})
            speed = ann.get("current_speed_cm_s") or cond.get("current_speed_cm_s")
            if speed is not None and speed != "":
                speed_val = float(speed)
                speeds.append(speed_val)
                amp = cond.get("wave_amplitude_cm") or ann.get("wave_amplitude_cm")
                if amp is not None and amp != "":
                    wave_scatter.append({"amplitude": float(amp), "speed": speed_val})
            curve = ann.get("cable_curvature_index")
            if curve is not None and curve != "":
                curvatures.append(float(curve))
            ang = ann.get("cable_angle_deg")
            if ang is not None and ang != "":
                angles.append(float(ang))
            angle = (cond.get("camera_angle") or ann.get("camera_angle") or "—")
            camera_angles[angle] = camera_angles.get(angle, 0) + 1
            dir_ = (cond.get("current_direction") or ann.get("current_direction") or "—")
            directions[dir_] = directions.get(dir_, 0) + 1
            pts = len(ann.get("points", []))
            point_counts.append(pts)
            author = (cond.get("annotator_name") or ann.get("annotator_name") or "Anonyme")
            annotators[author] = annotators.get(author, 0) + 1

    def make_hist(data, label_prefix=""):
        if not data: return []
        bins = np.linspace(min(data), max(data) + 1, 8)
        hist, bin_edges = np.histogram(data, bins=bins)
        return [{"range": f"{label_prefix}{int(bin_edges[i])}-{int(bin_edges[i+1])}", "count": int(hist[i])} for i in range(len(hist))]

    return {
        "total": total, "annotated": annotated_count, "ignored": ignored_count,
        "remaining": total - annotated_count - ignored_count,
        "speed_histogram": make_hist(speeds),
        "curvature_histogram": make_hist(curvatures),
        "angle_histogram": make_hist(angles, "θ="),
        "camera_angles": [{"name": k, "value": v} for k, v in camera_angles.items()],
        "current_directions": [{"name": k, "value": v} for k, v in directions.items()],
        "wave_scatter": wave_scatter,
        "avg_points": round(np.mean(point_counts), 1) if point_counts else 0,
        "annotators": [{"name": k, "count": v} for k, v in annotators.items()],
        "balance_warnings": [f"Valeur '{k}' peu représentée (<5)" for k, v in camera_angles.items() if v < 5 and k != "—"]
    }


# ─── Export ───────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/export/stats")
async def export_stats(name: str):
    meta = load_session_meta(name)
    annotated = [img for img in meta["images"] if img.get("status") == "annotated"]
    return {"total_annotated": len(annotated), "total_images": len(meta["images"]),
            "train_count": int(len(annotated) * 0.8), "val_count": len(annotated) - int(len(annotated) * 0.8)}

@app.get("/api/sessions/{name}/export/download")
async def export_download(name: str):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    annotated = [img for img in meta["images"] if img.get("status") == "annotated"]
    if not annotated:
        raise HTTPException(status_code=400, detail="No annotated images")
    random.seed(42)
    shuffled = annotated.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_imgs, val_imgs = shuffled[:split_idx], shuffled[split_idx:]
    if not train_imgs and val_imgs:
        train_imgs = [val_imgs.pop(0)]
    csv_header = ["filename", "split", "cable_angle_deg", "cable_angle_chord_deg",
                  "cable_curvature_index", "current_speed_cm_s", "annotator_name",
                  "current_direction", "wave_amplitude_cm", "wave_frequency_hz",
                  "wind_speed_m_s", "camera_angle", "water_turbidity",
                  "lighting_condition", "immersed_length_cm", "buoy_to_surface_cm",
                  "canal_water_depth_cm", "notes"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset/dataset.yaml", "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['mooring_line']\n")
        zf.writestr("dataset/train_yolo.py", 'from ultralytics import YOLO\nimport os\nos.chdir(os.path.dirname(os.path.abspath(__file__)))\nmodel = YOLO("yolov8n-seg.pt")\nmodel.train(data="dataset.yaml", epochs=100, imgsz=640, batch=8, name="cosmer_run")\n')
        csv_rows = []
        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            for img in imgs:
                filename = img["filename"]
                stem = Path(filename).stem
                img_src = session_dir / "images" / filename
                label_src = session_dir / "labels" / f"{stem}.txt"
                if img_src.exists():
                    zf.write(img_src, f"dataset/images/{split}/{filename}")
                if label_src.exists():
                    zf.write(label_src, f"dataset/labels/{split}/{stem}.txt")
                ann_path = session_dir / "annotations" / f"{stem}.json"
                row = {"filename": filename, "split": split}
                if ann_path.exists():
                    with open(ann_path) as af:
                        ann = json.load(af)
                    cond = ann.get("conditions", {})
                    for key in csv_header[2:]:
                        row[key] = ann.get(key, cond.get(key, ""))
                csv_rows.append(row)
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)
        zf.writestr("dataset/dataset_summary.csv", csv_buf.getvalue())
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=dataset_{name}.zip"})

@app.get("/api/export/global/stats")
async def global_export_stats(sessions: Optional[str] = Query(None)):
    selected = sessions.split(",") if sessions else None
    total_ann, total_img = 0, 0
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "session.json").exists():
            if selected and d.name not in selected:
                continue
            meta = load_session_meta(d.name)
            total_img += len(meta["images"])
            total_ann += sum(1 for img in meta["images"] if img.get("status") == "annotated")
    return {"total_annotated": total_ann, "total_images": total_img,
            "train_count": int(total_ann * 0.8), "val_count": total_ann - int(total_ann * 0.8)}

@app.get("/api/export/global/download")
async def global_export_download(sessions: Optional[str] = Query(None)):
    selected = sessions.split(",") if sessions else None
    all_annotated = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "session.json").exists():
            if selected and d.name not in selected:
                continue
            try:
                meta = load_session_meta(d.name)
                for img in meta["images"]:
                    if img.get("status") == "annotated":
                        all_annotated.append((d.name, img))
            except Exception:
                pass
    if not all_annotated:
        raise HTTPException(status_code=400, detail="No annotated images")
    random.seed(42)
    random.shuffle(all_annotated)
    split_idx = max(1, int(len(all_annotated) * 0.8))
    train_items, val_items = all_annotated[:split_idx], all_annotated[split_idx:]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("dataset/dataset.yaml", "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['mooring_line']\n")
        zf.writestr("dataset/train_yolo.py", 'from ultralytics import YOLO\nimport os\nos.chdir(os.path.dirname(os.path.abspath(__file__)))\nmodel = YOLO("yolov8n-seg.pt")\nmodel.train(data="dataset.yaml", epochs=100, imgsz=640, batch=8, name="cosmer_run")\n')
        csv_rows = []
        csv_header = ["session", "filename", "split", "cable_angle_deg",
                      "cable_angle_chord_deg", "cable_curvature_index",
                      "current_speed_cm_s", "annotator_name", "current_direction",
                      "wave_amplitude_cm", "wave_frequency_hz", "wind_speed_m_s",
                      "camera_angle", "water_turbidity", "lighting_condition",
                      "immersed_length_cm", "buoy_to_surface_cm", "canal_water_depth_cm", "notes"]
        for split_name, items in [("train", train_items), ("val", val_items)]:
            for session_name, img in items:
                sdir = get_session_dir(session_name)
                stem = Path(img["filename"]).stem
                dest_name = f"{session_name}__{img['filename']}"
                img_src = sdir / "images" / img["filename"]
                label_src = sdir / "labels" / f"{stem}.txt"
                if img_src.exists():
                    zf.write(img_src, f"dataset/images/{split_name}/{dest_name}")
                if label_src.exists():
                    zf.write(label_src, f"dataset/labels/{split_name}/{session_name}__{stem}.txt")
                ann_path = sdir / "annotations" / f"{stem}.json"
                row = {"session": session_name, "filename": img["filename"], "split": split_name}
                if ann_path.exists():
                    with open(ann_path) as af:
                        ann = json.load(af)
                    cond = ann.get("conditions", {})
                    for key in csv_header[3:]:
                        row[key] = ann.get(key, cond.get(key, ""))
                else:
                    for key in csv_header[3:]:
                        row[key] = ""
                csv_rows.append(row)
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(csv_rows)
        zf.writestr("dataset/annotations.csv", csv_buf.getvalue())
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": "attachment; filename=global_dataset.zip"})


# ─── AI Training ───────────────────────────────────────────────────────────────

def collect_training_data(session_names=None):
    rows = []
    sessions_dirs = {}
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not (d / "session.json").exists():
            continue
        if session_names and d.name not in session_names:
            continue
        images_dir = d / "images"
        sessions_dirs[d.name] = images_dir
        for ann_file in (d / "annotations").glob("*.json"):
            with open(ann_file) as f:
                ann = json.load(f)
            speed = ann.get("current_speed_cm_s") or ann.get("conditions", {}).get("current_speed_cm_s")
            if speed is None or speed == "":
                continue
            try:
                speed = float(speed)
            except (ValueError, TypeError):
                continue
            stem = ann_file.stem
            for ext in [".jpg", ".jpeg", ".png"]:
                img_file = images_dir / f"{stem}{ext}"
                if img_file.exists():
                    rows.append({"filename": img_file.name, "session_name": d.name,
                                 "current_speed_cm_s": speed})
                    break
    return rows, sessions_dirs


def _run_training(job_key: str, train_rows: list, val_rows: list,
                  sessions_dirs: dict, model_name: str, epochs: int,
                  progress_dict: dict):
    torch, nn, Dataset, DataLoader, transforms, models_tv, Image = get_torch_modules()
    if torch is None:
        progress_dict[job_key] = {"status": "error", "error": "PyTorch non installé. pip install torch torchvision"}
        return

    try:
        if not train_rows:
            progress_dict[job_key] = {"status": "error", "error": "Aucune image annotée avec current_speed_cm_s"}
            return

        # Dataset class (inline, hérite correctement de Dataset)
        class _CableDS(Dataset):
            def __init__(self, rows, sdirs, transform=None):
                self.rows = rows
                self.sdirs = sdirs
                self.transform = transform

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, idx):
                row = self.rows[idx]
                img_path = self.sdirs[row["session_name"]] / row["filename"]
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                speed = torch.tensor(row["current_speed_cm_s"], dtype=torch.float32)
                return img, speed

        T_TRAIN = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        T_VAL = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_ds = _CableDS(train_rows, sessions_dirs, T_TRAIN)
        val_ds   = _CableDS(val_rows,   sessions_dirs, T_VAL)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

        # ResNet18 — compatible torchvision >=0.13 ET <0.13
        try:
            model = models_tv.resnet18(weights=models_tv.ResNet18_Weights.DEFAULT)
        except AttributeError:
            model = models_tv.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model = model.to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        progress_dict[job_key] = {
            "epoch": 0, "total_epochs": epochs,
            "train_losses": [], "val_losses": [],
            "status": "running",
            "n_train": len(train_ds), "n_val": len(val_ds),
            "device": str(DEVICE),
        }

        preds_all, true_all = [], []
        for epoch in range(epochs):
            model.train()
            tl = 0.0
            for imgs, speeds in train_loader:
                imgs, speeds = imgs.to(DEVICE), speeds.to(DEVICE)
                optimizer.zero_grad()
                out = model(imgs).squeeze(-1)
                loss = criterion(out, speeds)
                loss.backward()
                optimizer.step()
                tl += loss.item() * imgs.size(0)
            tl /= len(train_ds)

            model.eval()
            vl = 0.0
            ep_preds, ep_true = [], []
            with torch.no_grad():
                for imgs, speeds in val_loader:
                    imgs, speeds = imgs.to(DEVICE), speeds.to(DEVICE)
                    out = model(imgs).squeeze(-1)
                    vl += criterion(out, speeds).item() * imgs.size(0)
                    ep_preds.extend(out.cpu().tolist())
                    ep_true.extend(speeds.cpu().tolist())
            vl = vl / len(val_ds) if len(val_ds) > 0 else 0.0
            scheduler.step(vl)
            preds_all, true_all = ep_preds, ep_true

            progress_dict[job_key]["epoch"] = epoch + 1
            progress_dict[job_key]["train_losses"].append(round(tl, 4))
            progress_dict[job_key]["val_losses"].append(round(vl, 4))

        preds_arr = np.array(preds_all)
        true_arr  = np.array(true_all)
        mae  = float(np.mean(np.abs(preds_arr - true_arr))) if len(preds_arr) else 0.0
        rmse = float(np.sqrt(np.mean((preds_arr - true_arr) ** 2))) if len(preds_arr) else 0.0

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(MODELS_DIR / f"{model_name}.pth"))

        progress_dict[job_key].update({
            "status": "done",
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "preds": preds_all,
            "true": true_all,
            "model_name": model_name,
        })
    except Exception as e:
        import traceback
        progress_dict[job_key] = {"status": "error", "error": str(e),
                                  "traceback": traceback.format_exc()}


@app.post("/api/sessions/{name}/train")
async def train_session_model(name: str, background_tasks: BackgroundTasks, epochs: int = Form(50)):
    session_dir = get_session_dir(name)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    torch_mods = get_torch_modules()
    if torch_mods[0] is None:
        raise HTTPException(status_code=503, detail="PyTorch non installé. pip install torch torchvision")
    rows, sessions_dirs = collect_training_data(session_names=[name])
    if not rows:
        raise HTTPException(status_code=400, detail="Aucune image annotée avec current_speed_cm_s")
    random.seed(42)
    random.shuffle(rows)
    split_idx = max(1, int(len(rows) * 0.8))
    train_rows, val_rows = rows[:split_idx], rows[split_idx:]
    model_name = f"{name}_vc_model"
    TRAIN_PROGRESS[name] = {"epoch": 0, "total_epochs": epochs, "status": "starting"}
    background_tasks.add_task(_run_training, name, train_rows, val_rows,
                              sessions_dirs, model_name, epochs, TRAIN_PROGRESS)
    return {"status": "started", "model_name": model_name, "n_images": len(rows)}


@app.post("/api/train/global")
async def train_global_model(
    background_tasks: BackgroundTasks,
    sessions: Optional[str] = Form(None),
    epochs: int = Form(50),
    model_name: str = Form("global_vc_model")
):
    torch_mods = get_torch_modules()
    if torch_mods[0] is None:
        raise HTTPException(status_code=503, detail="PyTorch non installé. pip install torch torchvision")
    session_names = [s.strip() for s in sessions.split(",")] if sessions else None
    rows, sessions_dirs = collect_training_data(session_names=session_names)
    if not rows:
        raise HTTPException(status_code=400, detail="Aucune image annotée avec current_speed_cm_s")
    random.seed(42)
    random.shuffle(rows)
    split_idx = max(1, int(len(rows) * 0.8))
    train_rows, val_rows = rows[:split_idx], rows[split_idx:]
    GLOBAL_TRAIN_PROGRESS["global"] = {"epoch": 0, "total_epochs": epochs, "status": "starting"}
    background_tasks.add_task(_run_training, "global", train_rows, val_rows,
                              sessions_dirs, model_name, epochs, GLOBAL_TRAIN_PROGRESS)
    return {
        "status": "started",
        "model_name": model_name,
        "n_sessions": len(set(r["session_name"] for r in rows)),
        "n_images": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
    }


@app.get("/api/sessions/{name}/train/progress")
async def get_session_train_progress(name: str):
    return TRAIN_PROGRESS.get(name, {"epoch": 0, "total_epochs": 50, "status": "idle"})

@app.get("/api/train/global/progress")
async def get_global_train_progress():
    return GLOBAL_TRAIN_PROGRESS.get("global", {"status": "idle"})


@app.get("/api/models")
async def list_models():
    if not MODELS_DIR.exists():
        return []
    result = []
    for f in sorted(MODELS_DIR.glob("*.pth")):
        result.append({
            "name": f.stem,
            "filename": f.name,
            "size_mb": round(f.stat().st_size / 1e6, 1),
            "modified_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "is_global": "global" in f.stem,
        })
    return sorted(result, key=lambda x: (not x["is_global"], x["modified_at"]), reverse=True)


@app.post("/api/predict")
async def predict_endpoint(model_name: str = Form(...), file: UploadFile = File(...)):
    torch, nn, _, _, transforms, models_tv, Image = get_torch_modules()
    if torch is None:
        raise HTTPException(status_code=503, detail="PyTorch non installé. pip install torch torchvision")
    model_path = MODELS_DIR / f"{model_name}.pth"
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Modèle {model_name} introuvable")
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = models_tv.resnet18(weights=None)
        except Exception:
            model = models_tv.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
        model.eval().to(DEVICE)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vc = model(tensor).item()
        return {"vitesse_estimee": round(vc, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
