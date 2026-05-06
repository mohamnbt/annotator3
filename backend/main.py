"""
COSMER Annotator — Backend FastAPI
Laboratoire COSMER, Université de Toulon
"""

import os
import re
import json
import csv
import io
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


# ─── Global exception handler ───────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────
def sanitize_name(name: str) -> str:
    """Sanitize session name: remove accents, special chars, replace spaces."""
    # Remove accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    # Replace spaces with underscores
    ascii_str = ascii_str.replace(" ", "_")
    # Keep only allowed chars
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
    """Write YOLO segmentation label file."""
    labels_dir = get_session_dir(session_name) / "labels"
    labels_dir.mkdir(exist_ok=True)
    label_path = labels_dir / f"{stem}.txt"
    
    if not points or len(points) < 2:
        return
    
    # Normalize coordinates
    normalized = []
    for pt in points:
        x_norm = pt["x"] / img_width
        y_norm = pt["y"] / img_height
        # Clamp to [0, 1]
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        normalized.append(f"{x_norm:.6f} {y_norm:.6f}")
    
    line = "0 " + " ".join(normalized) + "\n"
    with open(label_path, "w") as f:
        f.write(line)


# ─── Session Routes ─────────────────────────────────────────────────────────

@app.get("/api/sessions")
async def list_sessions():
    """List all sessions."""
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
    """Create a new session."""
    sanitized = sanitize_name(name)
    if not sanitized:
        raise HTTPException(status_code=400, detail="Invalid session name")
    
    session_dir = get_session_dir(sanitized)
    if session_dir.exists():
        raise HTTPException(status_code=409, detail=f"Session '{sanitized}' already exists")
    
    # Create directories
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
    """Get session details."""
    return load_session_meta(name)


@app.delete("/api/sessions/{name}")
async def delete_session(name: str):
    """Delete a session and all its data."""
    session_dir = get_session_dir(name)
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    shutil.rmtree(session_dir)
    return {"message": f"Session '{name}' deleted"}


@app.get("/api/sanitize")
async def sanitize_name_endpoint(name: str = Query(...)):
    """Preview sanitized session name."""
    return {"sanitized": sanitize_name(name)}


# ─── Image Routes ───────────────────────────────────────────────────────────

@app.post("/api/sessions/{name}/images")
async def upload_images(name: str, files: List[UploadFile] = File(...)):
    """Upload images to a session."""
    meta = load_session_meta(name)
    images_dir = get_session_dir(name) / "images"
    
    uploaded = []
    for file in files:
        # Sanitize filename
        original = file.filename or "image.jpg"
        stem = Path(original).stem
        ext = Path(original).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        
        safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)
        filename = f"{safe_stem}{ext}"
        
        # Avoid duplicates
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
            "status": "to_annotate",  # to_annotate, annotated, ignored
            "added_at": datetime.now().isoformat(),
        }
        meta["images"].append(img_entry)
        uploaded.append(filename)
    
    save_session_meta(name, meta)
    return {"uploaded": uploaded, "count": len(uploaded)}


@app.get("/api/sessions/{name}/images/{filename}")
async def get_image(name: str, filename: str):
    """Serve an image file."""
    filepath = get_session_dir(name) / "images" / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    
    ext = filepath.suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    
    return StreamingResponse(open(filepath, "rb"), media_type=media_type)


@app.delete("/api/sessions/{name}/images/{filename}")
async def delete_image(name: str, filename: str):
    """Delete an image and its associated annotation/label."""
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    
    # Remove image file
    img_path = session_dir / "images" / filename
    if img_path.exists():
        os.remove(img_path)
    
    stem = Path(filename).stem
    
    # Remove annotation
    ann_path = session_dir / "annotations" / f"{stem}.json"
    if ann_path.exists():
        os.remove(ann_path)
    
    # Remove label
    label_path = session_dir / "labels" / f"{stem}.txt"
    if label_path.exists():
        os.remove(label_path)
    
    # Remove from meta
    meta["images"] = [img for img in meta["images"] if img["filename"] != filename]
    save_session_meta(name, meta)
    
    return {"message": f"Image '{filename}' deleted"}


# ─── Video Extraction ───────────────────────────────────────────────────────

def get_video_rotation(video_path: str) -> int:
    """Read rotation metadata from video using ffprobe."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(video_path)
        ], capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        
        for stream in data.get("streams", []):
            # Check tags
            tags = stream.get("tags", {})
            if "rotate" in tags or "rotation" in tags:
                rotate = tags.get("rotate", tags.get("rotation", "0"))
                return int(float(str(rotate).strip()))
                
            # Check side_data_list (ffmpeg >= 5.0)
            for side_data in stream.get("side_data_list", []):
                if "rotation" in side_data:
                    return int(float(str(side_data["rotation"]).strip()))
    except Exception:
        pass
    return 0

def fix_frame_rotation(frame, rotation: int):
    """Apply inverse rotation to correct frame orientation."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270 or rotation == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

@app.post("/api/sessions/{name}/video")
async def extract_video_frames(
    name: str,
    file: UploadFile = File(...),
    frame_interval: int = Form(240),
):
    """Extract frames from a video at given interval."""
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    images_dir = session_dir / "images"
    
    # Save video temporarily
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
        
        return {
            "extracted": extracted,
            "total_video_frames": total_frames,
            "fps": fps,
            "frame_interval": frame_interval,
        }
    
    finally:
        if tmp_video.exists():
            os.remove(tmp_video)


# ─── Annotation Routes ──────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/annotations/{stem}")
async def get_annotation(name: str, stem: str):
    """Get annotation for an image."""
    ann_path = get_session_dir(name) / "annotations" / f"{stem}.json"
    if not ann_path.exists():
        return None
    
    with open(ann_path, "r") as f:
        return json.load(f)


@app.post("/api/sessions/{name}/annotations/{stem}")
async def save_annotation(name: str, stem: str, data: dict):
    """Save annotation and generate YOLO label."""
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    
    # Save annotation JSON
    ann_dir = session_dir / "annotations"
    ann_dir.mkdir(exist_ok=True)
    ann_path = ann_dir / f"{stem}.json"
    
    data["saved_at"] = datetime.now().isoformat()
    
    with open(ann_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Generate YOLO label
    points = data.get("points", [])
    img_width = data.get("image_width", 1)
    img_height = data.get("image_height", 1)
    
    # Handle contour mode (two polylines → polygon)
    mode = data.get("annotation_mode", "centerline")
    if mode == "contour" and "left_points" in data and "right_points" in data:
        # Combine left + reversed right to form a closed polygon
        left = data["left_points"]
        right = list(reversed(data["right_points"]))
        all_points = left + right
        write_yolo_label(name, stem, all_points, img_width, img_height)
    else:
        write_yolo_label(name, stem, points, img_width, img_height)
    
    # Update status in meta
    for img in meta["images"]:
        if Path(img["filename"]).stem == stem:
            img["status"] = "annotated"
            break
    
    save_session_meta(name, meta)
    
    return {"message": "Annotation saved", "stem": stem}


@app.post("/api/sessions/{name}/images/{filename}/ignore")
async def ignore_image(name: str, filename: str):
    """Mark image as ignored."""
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
    """Get conditions from the most recently annotated image."""
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    
    # Find last annotated image
    annotated = [img for img in meta["images"] if img.get("status") == "annotated"]
    if not annotated:
        return None
    
    # Get the most recently saved annotation
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


# ─── Export ──────────────────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/export/stats")
async def export_stats(name: str):
    """Get pre-export statistics."""
    meta = load_session_meta(name)
    
    annotated = [img for img in meta["images"] if img.get("status") == "annotated"]
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
    """Generate and download the dataset ZIP."""
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    
    annotated = [img for img in meta["images"] if img.get("status") == "annotated"]
    if not annotated:
        raise HTTPException(status_code=400, detail="No annotated images to export")
    
    # Shuffle and split 80/20
    random.seed(42)
    shuffled = annotated.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * 0.8)
    train_imgs = shuffled[:split_idx] if split_idx > 0 else shuffled
    val_imgs = shuffled[split_idx:] if split_idx < len(shuffled) else []
    
    # Ensure at least 1 in train
    if not train_imgs and val_imgs:
        train_imgs = [val_imgs.pop(0)]
    
    buf = io.BytesIO()
    
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # dataset.yaml
        yaml_content = """path: .
train: images/train
val: images/val
nc: 1
names: ['mooring_line']
"""
        zf.writestr("dataset/dataset.yaml", yaml_content)
        
        # train_yolo.py
        train_script = '''from ultralytics import YOLO
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = YOLO("yolov8n-seg.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640, batch=8, name="cosmer_run")
'''
        zf.writestr("dataset/train_yolo.py", train_script)
        
        # CSV header
        csv_rows = []
        csv_header = [
            "filename", "split", "annotator_name", "current_speed_cm_s",
            "current_direction", "wave_amplitude_cm", "wave_frequency_hz",
            "wind_speed_m_s", "camera_angle", "water_turbidity",
            "lighting_condition", "immersed_length_cm", "buoy_to_surface_cm",
            "canal_water_depth_cm", "notes", "num_points"
        ]
        
        def add_images(images, split_name):
            for img in images:
                filename = img["filename"]
                stem = Path(filename).stem
                
                # Image
                img_path = session_dir / "images" / filename
                if img_path.exists():
                    zf.write(img_path, f"dataset/images/{split_name}/{filename}")
                
                # Label
                label_path = session_dir / "labels" / f"{stem}.txt"
                if label_path.exists():
                    zf.write(label_path, f"dataset/labels/{split_name}/{stem}.txt")
                
                # Metadata
                ann_path = session_dir / "annotations" / f"{stem}.json"
                conditions = {}
                num_points = 0
                if ann_path.exists():
                    with open(ann_path, "r") as f:
                        ann = json.load(f)
                    conditions = ann.get("conditions", {})
                    num_points = len(ann.get("points", []))
                    zf.writestr(
                        f"dataset/metadata/{stem}_meta.json",
                        json.dumps(ann, indent=2, ensure_ascii=False)
                    )
                
                csv_rows.append({
                    "filename": filename,
                    "split": split_name,
                    "annotator_name": conditions.get("annotator_name", ""),
                    "current_speed_cm_s": conditions.get("current_speed_cm_s", ""),
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
        
        # CSV
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=csv_header)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
        zf.writestr("dataset/dataset_summary.csv", csv_buf.getvalue())
    
    buf.seek(0)
    
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=cosmer_dataset_{name}.zip"
        },
    )


# ─── Statistics ──────────────────────────────────────────────────────────────

@app.get("/api/sessions/{name}/statistics")
async def get_statistics(name: str):
    """Get session statistics for dashboard."""
    meta = load_session_meta(name)
    session_dir = get_session_dir(name)
    
    total = len(meta["images"])
    annotated = sum(1 for img in meta["images"] if img.get("status") == "annotated")
    ignored = sum(1 for img in meta["images"] if img.get("status") == "ignored")
    remaining = total - annotated - ignored
    
    # Collect all conditions
    speeds = []
    camera_angles = {}
    current_directions = {}
    wave_data = []  # (amplitude, speed)
    point_counts = []
    annotators = {}
    condition_counts = {}  # for balance warning
    
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
        
        # Speed
        speed = conditions.get("current_speed_cm_s")
        if speed is not None and speed != "":
            try:
                speeds.append(float(speed))
            except (ValueError, TypeError):
                pass
        
        # Camera angle
        angle = conditions.get("camera_angle", "")
        if angle and angle != "—":
            camera_angles[angle] = camera_angles.get(angle, 0) + 1
            key = f"camera_angle:{angle}"
            condition_counts[key] = condition_counts.get(key, 0) + 1
        
        # Current direction
        direction = conditions.get("current_direction", "")
        if direction and direction != "—":
            current_directions[direction] = current_directions.get(direction, 0) + 1
            key = f"current_direction:{direction}"
            condition_counts[key] = condition_counts.get(key, 0) + 1
        
        # Wave data
        wave_amp = conditions.get("wave_amplitude_cm")
        wave_speed = conditions.get("current_speed_cm_s")
        if wave_amp is not None and wave_speed is not None:
            try:
                wave_data.append({
                    "amplitude": float(wave_amp),
                    "speed": float(wave_speed),
                })
            except (ValueError, TypeError):
                pass
        
        # Annotator
        annotator = conditions.get("annotator_name", "Anonyme")
        if not annotator:
            annotator = "Anonyme"
        annotators[annotator] = annotators.get(annotator, 0) + 1
    
    # Balance warnings
    balance_warnings = []
    for key, count in condition_counts.items():
        if count < 5:
            balance_warnings.append(f"{key}: {count} échantillons")
    
    # Speed histogram (bins)
    speed_histogram = []
    if speeds:
        min_s = min(speeds)
        max_s = max(speeds)
        n_bins = min(10, len(set(speeds)))
        if n_bins > 0 and max_s > min_s:
            bin_width = (max_s - min_s) / n_bins
            for i in range(n_bins):
                lo = min_s + i * bin_width
                hi = lo + bin_width
                count = sum(1 for s in speeds if lo <= s < hi or (i == n_bins - 1 and s == hi))
                speed_histogram.append({
                    "range": f"{lo:.1f}-{hi:.1f}",
                    "count": count,
                })
    
    return {
        "total": total,
        "annotated": annotated,
        "ignored": ignored,
        "remaining": remaining,
        "speed_histogram": speed_histogram,
        "camera_angles": [{"name": k, "value": v} for k, v in camera_angles.items()],
        "current_directions": [{"name": k, "value": v} for k, v in current_directions.items()],
        "wave_scatter": wave_data,
        "avg_points": round(sum(point_counts) / len(point_counts), 1) if point_counts else 0,
        "point_counts": point_counts,
        "annotators": [{"name": k, "count": v} for k, v in annotators.items()],
        "balance_warnings": balance_warnings,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
