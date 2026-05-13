"""
COSMER Annotator — Backend FastAPI
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
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np

app = FastAPI(title="COSMER Annotator API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "sessions"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# YOLO model (chargé une seule fois au démarrage)
YOLO_MODEL = None
YOLO_MODEL_PATH = Path(__file__).parent / "best.pt"

def get_yolo_model():
    global YOLO_MODEL
    if YOLO_MODEL is None and YOLO_MODEL_PATH.exists():
        try:
            from ultralytics import YOLO
            YOLO_MODEL = YOLO(str(YOLO_MODEL_PATH))
        except Exception as e:
            print(f"[YOLO] Impossible de charger le modèle: {e}")
    return YOLO_MODEL

# Préchargement au démarrage
try:
    get_yolo_model()
except Exception:
    pass


# ─── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


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
    session_dir = get_session_dir(name)
    meta_path = session_dir / "session.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Session '{name}' not found")
    with open(meta_path, "r") as f:
        return json.load(f)

def save_session_meta(name: str, meta: dict):
    session_dir = get_session_dir(name)
    meta_path = session_dir / "session.json"
    with open(meta_path, "w") as f:
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


# ─── Calcul d'angle du câble ──────────────────────────────────────────────────

def calc_cable_angle(points: list) -> dict:
    """
    Calcule l'angle θ du câble par rapport à la verticale à partir d'une polyligne.

    Convention : θ = 0° quand le câble est parfaitement vertical (pas de courant),
                 θ augmente quand le câble s'incline sous l'effet du courant.
                 Plage attendue : 0° à ~40° pour Vc entre 5 et 30 cm/s.

    Méthode principale : Régression linéaire (moindres carrés) sur tous les points.
    - Donne l'orientation moyenne globale du câble, robuste au bruit de tracé.
    - Si câble droit → identique à la corde 2 points.
    - Si câble courbé → meilleur estimateur de l'angle "équivalent tige rigide".
    - Directement réutilisable sur des segments d'un câble long (futur travail en mer).

    Retourne aussi :
    - cable_angle_chord_deg : angle corde (point 1 → point N), pour comparaison.
    - cable_curvature_index : écart entre les deux angles, indicateur de courbure.
    """
    result = {
        "cable_angle_deg": "",
        "cable_angle_chord_deg": "",
        "cable_curvature_index": "",
    }

    if not points or len(points) < 2:
        return result

    xs = np.array([float(p["x"]) for p in points])
    ys = np.array([float(p["y"]) for p in points])

    # ── Angle par régression linéaire (méthode principale) ──
    # On cherche la direction principale par PCA (robuste aux câbles quasi-verticaux
    # ET quasi-horizontaux, contrairement à polyfit qui diverge si câble vertical).
    coords = np.stack([xs, ys], axis=1)
    mean = coords.mean(axis=0)
    centered = coords - mean
    # Matrice de covariance 2×2
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        # Cas dégénéré : tous les points alignés sur une droite parfaite
        cov = np.array([[float(cov), 0.0], [0.0, 0.0]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Vecteur directeur de la droite de régression (direction de variance maximale)
    principal = eigenvectors[:, np.argmax(eigenvalues)]  # [dx, dy] en coords pixels
    dx_reg = float(principal[0])
    dy_reg = float(principal[1])

    # Assurer que dy pointe vers le bas (ancrage en bas de l'image = y croissant)
    # Si le premier point est en haut (y petit) et le dernier en bas (y grand),
    # on oriente dy positif vers le bas pour que l'angle soit mesuré correctement.
    if (ys[-1] - ys[0]) < 0:
        dy_reg = -dy_reg
        dx_reg = -dx_reg

    # θ = angle entre la direction du câble et la verticale descendante (dy > 0)
    # atan2(|dx|, dy) → 0° si vertical, 90° si horizontal
    angle_reg = math.degrees(math.atan2(abs(dx_reg), abs(dy_reg)))
    result["cable_angle_deg"] = round(angle_reg, 3)

    # ── Angle corde (point 1 → point N) ──
    dx_chord = float(xs[-1] - xs[0])
    dy_chord = float(ys[0] - ys[-1])  # inversé car Y pixel croît vers le bas
    # Si les deux extrémités sont confondues, fallback sur régression
    if abs(dx_chord) < 1e-6 and abs(dy_chord) < 1e-6:
        result["cable_angle_chord_deg"] = result["cable_angle_deg"]
    else:
        angle_chord = math.degrees(math.atan2(abs(dx_chord), abs(dy_chord)))
        result["cable_angle_chord_deg"] = round(angle_chord, 3)

    # ── Indice de courbure ──
    try:
        curvature = round(abs(float(result["cable_angle_deg"]) - float(result["cable_angle_chord_deg"])), 3)
        result["cable_curvature_index"] = curvature
    except (TypeError, ValueError):
        result["cable_curvature_index"] = ""

    return result


def extract_centerline_from_mask(mask_xy: np.ndarray, img_w: int, img_h: int, n_points: int = 40) -> list:
    """
    Extrait la ligne centrale (squelette) d'un masque de segmentation.
    Retourne une liste ordonnée de points {x, y} le long de l'axe du câble.
    """
    # Rasteriser le polygone masque dans une image binaire
    mask_img = np.zeros((img_h, img_w), dtype=np.uint8)
    pts = mask_xy.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask_img, [pts], 255)

    # Squelettisation via distance transform + thinning
    try:
        from skimage.morphology import skeletonize
        binary = mask_img > 0
        skeleton = skeletonize(binary).astype(np.uint8) * 255
    except ImportError:
        # Fallback sans scikit-image : on utilise le contour médian
        skeleton = mask_img.copy()

    # Récupérer les pixels du squelette
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        # Fallback : centroïdes par tranches verticales/horizontales
        return _centerline_slices(mask_img, img_w, img_h, n_points)

    skeleton_pts = np.stack([xs, ys], axis=1).astype(float)

    # Ordonner les points du squelette du haut vers le bas (ou gauche→droite)
    # On utilise une projection PCA pour trouver l'axe principal
    mean = skeleton_pts.mean(axis=0)
    centered = skeleton_pts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    projections = centered @ principal_axis
    order = np.argsort(projections)
    ordered_pts = skeleton_pts[order]

    # Sous-échantillonner à n_points régulièrement espacés
    if len(ordered_pts) > n_points:
        indices = np.linspace(0, len(ordered_pts) - 1, n_points, dtype=int)
        ordered_pts = ordered_pts[indices]

    return [{"x": float(p[0]), "y": float(p[1])} for p in ordered_pts]


def _centerline_slices(mask_img: np.ndarray, img_w: int, img_h: int, n_points: int) -> list:
    """Fallback : centroïde par tranche pour estimer la ligne centrale."""
    points = []
    # Déterminer si le câble est plutôt vertical ou horizontal
    ys, xs = np.where(mask_img > 0)
    if len(xs) == 0:
        return []
    span_x = xs.max() - xs.min()
    span_y = ys.max() - ys.min()

    if span_y >= span_x:
        # Câble vertical : tranches horizontales
        y_min, y_max = int(ys.min()), int(ys.max())
        for i in range(n_points):
            y = int(y_min + (y_max - y_min) * i / (n_points - 1))
            row = np.where(mask_img[y, :] > 0)[0]
            if len(row) > 0:
                x = float(row.mean())
                points.append({"x": x, "y": float(y)})
    else:
        # Câble horizontal : tranches verticales
        x_min, x_max = int(xs.min()), int(xs.max())
        for i in range(n_points):
            x = int(x_min + (x_max - x_min) * i / (n_points - 1))
            col = np.where(mask_img[:, x] > 0)[0]
            if len(col) > 0:
                y = float(col.mean())
                points.append({"x": float(x), "y": y})
    return points


# ─── Session Routes ────────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    if not DATA_DIR.exists():
        return sessions
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "session.json").exists():
            meta = load_session_meta(d.name)
            sessions.append(meta)
    return sessions

@app.post("/api/sessions")
async def create_session(name: str = Form(...), description: str = Form("")):
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
        "created_at": datetime.now().isoformat(),
        "images": [],
    }
    save_session_meta(sanitized, meta)
    return meta

@app.get("/api/sessions/{name}")
async def get_session(name: str):
    return load_session_meta(name)

@app.delete("/api/sessions/{name}")
async def delete_session(name: str):
    session_dir = get_session_dir(name)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    shutil.rmtree(session_dir)
    return {"message": f"Session '{name}' deleted"}

@app.get("/api/sanitize")
async def sanitize_name_endpoint(name: str = Query(...)):
    return {"sanitized": sanitize_name(name)}


# ─── Image Routes ──────────────────────────────────────────────────────────────

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
        img_entry = {
            "filename": filename,
            "status": "to_annotate",
            "added_at": datetime.now().isoformat(),
        }
        meta["images"].append(img_entry)
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
    ann_path = session_dir / "annotations" / f"{stem}.json"
    if ann_path.exists():
        os.remove(ann_path)
    label_path = session_dir / "labels" / f"{stem}.txt"
    if label_path.exists():
        os.remove(label_path)
    meta["images"] = [img for img in meta["images"] if img["filename"] != filename]
    save_session_meta(name, meta)
    return {"message": f"Image '{filename}' deleted"}


# ─── YOLO Auto-annotation ──────────────────────────────────────────────────────

@app.get("/api/yolo/status")
async def yolo_status():
    """Vérifie si le modèle YOLO est disponible."""
    model_exists = YOLO_MODEL_PATH.exists()
    model_loaded = YOLO_MODEL is not None
    return {
        "model_path": str(YOLO_MODEL_PATH),
        "model_exists": model_exists,
        "model_loaded": model_loaded,
    }

@app.get("/api/sessions/{name}/images/{filename}/predict")
async def predict_annotation(name: str, filename: str, conf: float = Query(0.5)):
    """Lance l'inférence YOLO et retourne la ligne centrale du câble (squelette)."""
    model = get_yolo_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Modèle YOLO non disponible. Placez best.pt dans {YOLO_MODEL_PATH}"
        )

    img_path = get_session_dir(name) / "images" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        results = model.predict(str(img_path), conf=conf, verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks) == 0:
            return {"found": False, "points": [], "message": "Aucun objet détecté"}

        best_idx = int(result.boxes.conf.argmax())
        mask_xy = result.masks.xy[best_idx]  # contour du masque en pixels

        img_w = result.orig_shape[1]
        img_h = result.orig_shape[0]

        # Extraire la ligne centrale depuis le masque
        points = extract_centerline_from_mask(mask_xy, img_w, img_h, n_points=40)

        if not points:
            # Fallback : retourner le contour brut si squelette vide
            points = [{"x": float(pt[0]), "y": float(pt[1])} for pt in mask_xy]

        conf_val = float(result.boxes.conf[best_idx])
        return {
            "found": True,
            "points": points,
            "image_width": img_w,
            "image_height": img_h,
            "confidence": conf_val,
            "message": f"{len(points)} points (centerline, conf={conf_val:.2f})"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur inférence YOLO: {str(e)}")


# ─── Video Extraction ──────────────────────────────────────────────────────────

def get_video_rotation(video_path: str) -> int:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(video_path)],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            tags = stream.get("tags", {})
            if "rotate" in tags or "rotation" in tags:
                return int(float(str(tags.get("rotate", tags.get("rotation", "0"))).strip()))
            for side_data in stream.get("side_data_list", []):
                if "rotation" in side_data:
                    return int(float(str(side_data["rotation"]).strip()))
    except Exception:
        pass
    return 0

def fix_frame_rotation(frame, rotation: int):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270 or rotation == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

@app.post("/api/sessions/{name}/video")
async def extract_video_frames(name: str, file: UploadFile = File(...), frame_interval: int = Form(240)):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    images_dir = session_dir / "images"
    video_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(file.filename or "video").stem)
    tmp_video = session_dir / f"_tmp_{video_stem}.mp4"
    try:
        content = await file.read()
        with open(tmp_video, "wb") as f:
            f.write(content)
        cap = cv2.VideoCapture(str(tmp_video))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_idx = 0
        extracted = 0
        rotation = get_video_rotation(str(tmp_video))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = fix_frame_rotation(frame, rotation)
            if frame_idx % frame_interval == 0:
                filename = f"{video_stem}_frame_{frame_idx:06d}.jpg"
                filepath = images_dir / filename
                cv2.imwrite(str(filepath), frame)
                img_entry = {
                    "filename": filename,
                    "status": "to_annotate",
                    "added_at": datetime.now().isoformat(),
                    "source_video": file.filename,
                    "source_frame": frame_idx,
                }
                meta["images"].append(img_entry)
                extracted += 1
            frame_idx += 1
        cap.release()
        save_session_meta(name, meta)
        return {"extracted": extracted, "total_video_frames": total_frames, "fps": fps, "frame_interval": frame_interval}
    finally:
        if tmp_video.exists():
            os.remove(tmp_video)


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

    # Calcul automatique de l'angle au moment de la sauvegarde
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
    if latest and "conditions" in latest:
        return latest["conditions"]
    return None


# ─── Export (session individuelle) ────────────────────────────────────────────

@app.get("/api/sessions/{name}/export/stats")
async def export_stats(name: str):
    meta = load_session_meta(name)
    annotated = [img for img in meta["images"] if img.get("status") == "annotated"]
    total = len(annotated)
    train_count = int(total * 0.8)
    val_count = total - train_count
    return {"total_annotated": total, "total_images": len(meta["images"]), "train_count": train_count, "val_count": val_count}

@app.get("/api/sessions/{name}/export/download")
async def export_download(name: str):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    annotated = [img for img in meta["images"] if img.get("status") == "annotated"]
    if not annotated:
        raise HTTPException(status_code=400, detail="No annotated images to export")
    random.seed(42)
    shuffled = annotated.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.8)
    train_imgs = shuffled[:split_idx] if split_idx > 0 else shuffled
    val_imgs = shuffled[split_idx:] if split_idx < len(shuffled) else []
    if not train_imgs and val_imgs:
        train_imgs = [val_imgs.pop(0)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        yaml_content = "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['mooring_line']\n"
        zf.writestr("dataset/dataset.yaml", yaml_content)
        train_script = 'from ultralytics import YOLO\nimport os\n\nos.chdir(os.path.dirname(os.path.abspath(__file__)))\n\nmodel = YOLO("yolov8n-seg.pt")\nmodel.train(data="dataset.yaml", epochs=100, imgsz=640, batch=8, name="cosmer_run")\n'
        zf.writestr("dataset/train_yolo.py", train_script)
        csv_rows = []
        csv_header = [
            "filename", "split",
            "cable_angle_deg", "cable_angle_chord_deg", "cable_curvature_index",
            "current_speed_cm_s", "annotator_name", "current_direction",
            "wave_amplitude_cm", "wave_frequency_hz", "wind_speed_m_s",
            "camera_angle", "water_turbidity", "lighting_condition",
            "immersed_length_cm", "buoy_to_surface_cm", "canal_water_depth_cm",
            "notes", "num_points"
        ]
        def add_images(images, split_name):
            for img in images:
                filename = img["filename"]
                stem = Path(filename).stem
                img_path = session_dir / "images" / filename
                if img_path.exists():
                    zf.write(img_path, f"dataset/images/{split_name}/{filename}")
                label_path = session_dir / "labels" / f"{stem}.txt"
                if label_path.exists():
                    zf.write(label_path, f"dataset/labels/{split_name}/{stem}.txt")
                ann_path = session_dir / "annotations" / f"{stem}.json"
                conditions = {}
                num_points = 0
                cable_angle_deg = ""
                cable_angle_chord_deg = ""
                cable_curvature_index = ""
                if ann_path.exists():
                    with open(ann_path, "r") as f:
                        ann = json.load(f)
                    conditions = ann.get("conditions", {})
                    pts = ann.get("points", [])
                    num_points = len(pts)
                    # Récupère l'angle stocké, ou le recalcule à la volée pour les anciennes annotations
                    if "cable_angle_deg" in ann and ann["cable_angle_deg"] != "":
                        cable_angle_deg = ann["cable_angle_deg"]
                        cable_angle_chord_deg = ann.get("cable_angle_chord_deg", "")
                        cable_curvature_index = ann.get("cable_curvature_index", "")
                    elif pts and len(pts) >= 2:
                        angle_data = calc_cable_angle(pts)
                        cable_angle_deg = angle_data["cable_angle_deg"]
                        cable_angle_chord_deg = angle_data["cable_angle_chord_deg"]
                        cable_curvature_index = angle_data["cable_curvature_index"]
                    zf.writestr(f"dataset/metadata/{stem}_meta.json", json.dumps(ann, indent=2, ensure_ascii=False))
                csv_rows.append({
                    "filename": filename,
                    "split": split_name,
                    "cable_angle_deg": cable_angle_deg,
                    "cable_angle_chord_deg": cable_angle_chord_deg,
                    "cable_curvature_index": cable_curvature_index,
                    "current_speed_cm_s": conditions.get("current_speed_cm_s", ""),
                    "annotator_name": conditions.get("annotator_name", ""),
                    "current_direction": conditions.get("current_direction", ""),
                    "wave_amplitude_cm": conditions.get("wave_amplitude_cm", ""),
                    "wave_frequency_hz": conditions.get("wave_frequency_hz", ""),
                    "wind_speed_m_s": conditions.get("wind_speed_m_s", ""),
                    "camera_angle": conditions.get("camera_angle", ""),
                    "water_turbidity": conditions.get("water_turbidity", ""),
                    "lighting_condition": conditions.get("lighting_condition", ""),
                    "immersed_length_cm": conditions.get("immersed_length_cm", ""),
                    "buoy_to_surface_cm": conditions.get("buoy_to_surface_cm", ""),
                    "canal_water_depth_cm": conditions.get("canal_water_depth_cm", ""),
                    "notes": conditions.get("notes", ""),
                    "num_points": num_points,
                })
        add_images(train_imgs, "train")
        add_images(val_imgs, "val")
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=csv_header)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
        zf.writestr("dataset/dataset_summary.csv", csv_buf.getvalue())
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=cosmer_dataset_{name}.zip"})


# ─── Export fusionné (toutes sessions) ────────────────────────────────────────

@app.get("/api/export/merged/stats")
async def merged_export_stats():
    total_annotated = 0
    total_images = 0
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "session.json").exists():
            meta = load_session_meta(d.name)
            total_images += len(meta["images"])
            total_annotated += sum(1 for img in meta["images"] if img.get("status") == "annotated")
    train_count = int(total_annotated * 0.8)
    val_count = total_annotated - train_count
    return {"total_annotated": total_annotated, "total_images": total_images, "train_count": train_count, "val_count": val_count}

@app.get("/api/export/merged/download")
async def export_merged_download():
    all_annotated = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "session.json").exists():
            meta = load_session_meta(d.name)
            for img in meta["images"]:
                if img.get("status") == "annotated":
                    all_annotated.append((d.name, img))
    if not all_annotated:
        raise HTTPException(status_code=400, detail="Aucune image annotée trouvée")
    random.seed(42)
    shuffled = all_annotated.copy()
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * 0.8))
    train_items = shuffled[:split_idx]
    val_items = shuffled[split_idx:] if split_idx < len(shuffled) else []
    buf = io.BytesIO()
    csv_rows = []
    csv_header = [
        "filename", "session", "split",
        "cable_angle_deg", "cable_angle_chord_deg", "cable_curvature_index",
        "current_speed_cm_s", "annotator_name", "current_direction",
        "wave_amplitude_cm", "wave_frequency_hz", "wind_speed_m_s",
        "camera_angle", "water_turbidity", "lighting_condition",
        "immersed_length_cm", "buoy_to_surface_cm", "canal_water_depth_cm",
        "notes", "num_points"
    ]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        yaml_content = "path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['mooring_line']\n"
        zf.writestr("dataset/dataset.yaml", yaml_content)
        train_script = 'from ultralytics import YOLO\nimport os\n\nos.chdir(os.path.dirname(os.path.abspath(__file__)))\n\nmodel = YOLO("yolov8n-seg.pt")\nmodel.train(data="dataset.yaml", epochs=100, imgsz=640, batch=8, name="cosmer_run_merged")\n'
        zf.writestr("dataset/train_yolo.py", train_script)
        def add_merged_items(items, split_name):
            for session_name, img in items:
                session_dir = get_session_dir(session_name)
                filename = img["filename"]
                stem = Path(filename).stem
                ext = Path(filename).suffix
                unique_stem = f"{session_name}_{stem}"
                unique_filename = f"{unique_stem}{ext}"
                img_path = session_dir / "images" / filename
                if img_path.exists():
                    zf.write(img_path, f"dataset/images/{split_name}/{unique_filename}")
                label_path = session_dir / "labels" / f"{stem}.txt"
                if label_path.exists():
                    zf.write(label_path, f"dataset/labels/{split_name}/{unique_stem}.txt")
                ann_path = session_dir / "annotations" / f"{stem}.json"
                conditions = {}
                num_points = 0
                cable_angle_deg = ""
                cable_angle_chord_deg = ""
                cable_curvature_index = ""
                if ann_path.exists():
                    with open(ann_path, "r") as f:
                        ann = json.load(f)
                    conditions = ann.get("conditions", {})
                    pts = ann.get("points", [])
                    num_points = len(pts)
                    # Récupère l'angle stocké, ou le recalcule à la volée pour les anciennes annotations
                    if "cable_angle_deg" in ann and ann["cable_angle_deg"] != "":
                        cable_angle_deg = ann["cable_angle_deg"]
                        cable_angle_chord_deg = ann.get("cable_angle_chord_deg", "")
                        cable_curvature_index = ann.get("cable_curvature_index", "")
                    elif pts and len(pts) >= 2:
                        angle_data = calc_cable_angle(pts)
                        cable_angle_deg = angle_data["cable_angle_deg"]
                        cable_angle_chord_deg = angle_data["cable_angle_chord_deg"]
                        cable_curvature_index = angle_data["cable_curvature_index"]
                    zf.writestr(f"dataset/metadata/{unique_stem}_meta.json", json.dumps(ann, indent=2, ensure_ascii=False))
                csv_rows.append({
                    "filename": unique_filename,
                    "session": session_name,
                    "split": split_name,
                    "cable_angle_deg": cable_angle_deg,
                    "cable_angle_chord_deg": cable_angle_chord_deg,
                    "cable_curvature_index": cable_curvature_index,
                    "current_speed_cm_s": conditions.get("current_speed_cm_s", ""),
                    "annotator_name": conditions.get("annotator_name", ""),
                    "current_direction": conditions.get("current_direction", ""),
                    "wave_amplitude_cm": conditions.get("wave_amplitude_cm", ""),
                    "wave_frequency_hz": conditions.get("wave_frequency_hz", ""),
                    "wind_speed_m_s": conditions.get("wind_speed_m_s", ""),
                    "camera_angle": conditions.get("camera_angle", ""),
                    "water_turbidity": conditions.get("water_turbidity", ""),
                    "lighting_condition": conditions.get("lighting_condition", ""),
                    "immersed_length_cm": conditions.get("immersed_length_cm", ""),
                    "buoy_to_surface_cm": conditions.get("buoy_to_surface_cm", ""),
                    "canal_water_depth_cm": conditions.get("canal_water_depth_cm", ""),
                    "notes": conditions.get("notes", ""),
                    "num_points": num_points,
                })
        add_merged_items(train_items, "train")
        add_merged_items(val_items, "val")
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=csv_header)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
        zf.writestr("dataset/dataset_summary.csv", csv_buf.getvalue())
    buf.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=cosmer_dataset_merged_{timestamp}.zip"})


# ─── Statistics ────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/statistics")
async def get_statistics(name: str):
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    total = len(meta["images"])
    annotated = sum(1 for img in meta["images"] if img.get("status") == "annotated")
    ignored = sum(1 for img in meta["images"] if img.get("status") == "ignored")
    remaining = total - annotated - ignored
    speeds = []; camera_angles = {}; current_directions = {}; wave_data = []; point_counts = []; annotators = {}; condition_counts = {}
    for img in meta["images"]:
        if img.get("status") != "annotated":
            continue
        stem = Path(img["filename"]).stem
        ann_path = session_dir / "annotations" / f"{stem}.json"
        if not ann_path.exists():
            continue
        with open(ann_path, "r") as f:
            ann = json.load(f)
        conditions = ann.get("conditions", {})
        points = ann.get("points", [])
        point_counts.append(len(points))
        speed = conditions.get("current_speed_cm_s")
        if speed is not None and speed != "":
            try: speeds.append(float(speed))
            except (ValueError, TypeError): pass
        angle = conditions.get("camera_angle", "")
        if angle and angle != "—":
            camera_angles[angle] = camera_angles.get(angle, 0) + 1
            condition_counts[f"camera_angle:{angle}"] = condition_counts.get(f"camera_angle:{angle}", 0) + 1
        direction = conditions.get("current_direction", "")
        if direction and direction != "—":
            current_directions[direction] = current_directions.get(direction, 0) + 1
            condition_counts[f"current_direction:{direction}"] = condition_counts.get(f"current_direction:{direction}", 0) + 1
        wave_amp = conditions.get("wave_amplitude_cm")
        wave_speed = conditions.get("current_speed_cm_s")
        if wave_amp is not None and wave_speed is not None:
            try: wave_data.append({"amplitude": float(wave_amp), "speed": float(wave_speed)})
            except (ValueError, TypeError): pass
        annotator = conditions.get("annotator_name", "Anonyme") or "Anonyme"
        annotators[annotator] = annotators.get(annotator, 0) + 1
    balance_warnings = [f"{k}: {v} échantillons" for k, v in condition_counts.items() if v < 5]
    speed_histogram = []
    if speeds:
        min_s, max_s = min(speeds), max(speeds)
        n_bins = min(10, len(set(speeds)))
        if n_bins > 0 and max_s > min_s:
            bin_width = (max_s - min_s) / n_bins
            for i in range(n_bins):
                lo = min_s + i * bin_width
                hi = lo + bin_width
                count = sum(1 for s in speeds if lo <= s < hi or (i == n_bins - 1 and s == hi))
                speed_histogram.append({"range": f"{lo:.1f}-{hi:.1f}", "count": count})
    return {"total": total, "annotated": annotated, "ignored": ignored, "remaining": remaining, "speed_histogram": speed_histogram, "camera_angles": [{"name": k, "value": v} for k, v in camera_angles.items()], "current_directions": [{"name": k, "value": v} for k, v in current_directions.items()], "wave_scatter": wave_data, "avg_points": round(sum(point_counts) / len(point_counts), 1) if point_counts else 0, "point_counts": point_counts, "annotators": [{"name": k, "count": v} for k, v in annotators.items()], "balance_warnings": balance_warnings}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
