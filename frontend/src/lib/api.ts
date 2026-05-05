const API = "http://localhost:8000";

export interface SessionImage {
  filename: string;
  status: "to_annotate" | "annotated" | "ignored";
  added_at: string;
  source_video?: string;
  source_frame?: number;
}

export interface SessionMeta {
  name: string;
  description: string;
  created_at: string;
  images: SessionImage[];
}

export interface Conditions {
  annotator_name?: string;
  current_speed_cm_s?: number | string;
  current_direction?: string;
  wave_amplitude_cm?: number | string;
  wave_frequency_hz?: number | string;
  wind_speed_m_s?: number | string;
  camera_angle?: string;
  water_turbidity?: string;
  lighting_condition?: string;
  immersed_length_cm?: number | string;
  buoy_to_surface_cm?: number | string;
  canal_water_depth_cm?: number | string;
  notes?: string;
}

export interface AnnotationData {
  points: { x: number; y: number }[];
  left_points?: { x: number; y: number }[];
  right_points?: { x: number; y: number }[];
  annotation_mode: string;
  conditions: Conditions;
  image_width: number;
  image_height: number;
  saved_at?: string;
}

export interface ExportStats {
  total_annotated: number;
  total_images: number;
  train_count: number;
  val_count: number;
}

export interface Statistics {
  total: number;
  annotated: number;
  ignored: number;
  remaining: number;
  speed_histogram: { range: string; count: number }[];
  camera_angles: { name: string; value: number }[];
  current_directions: { name: string; value: number }[];
  wave_scatter: { amplitude: number; speed: number }[];
  avg_points: number;
  point_counts: number[];
  annotators: { name: string; count: number }[];
  balance_warnings: string[];
}

// ─── Sessions ────────────────────────────────────────────────────────────

export async function listSessions(): Promise<SessionMeta[]> {
  const res = await fetch(`${API}/api/sessions`);
  if (!res.ok) throw new Error("Failed to list sessions");
  return res.json();
}

export async function createSession(name: string, description: string): Promise<SessionMeta> {
  const form = new FormData();
  form.append("name", name);
  form.append("description", description);
  const res = await fetch(`${API}/api/sessions`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Failed to create session");
  }
  return res.json();
}

export async function getSession(name: string): Promise<SessionMeta> {
  const res = await fetch(`${API}/api/sessions/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error("Session not found");
  return res.json();
}

export async function deleteSession(name: string): Promise<void> {
  const res = await fetch(`${API}/api/sessions/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete session");
}

export async function sanitizeName(name: string): Promise<string> {
  const res = await fetch(`${API}/api/sanitize?name=${encodeURIComponent(name)}`);
  const data = await res.json();
  return data.sanitized;
}

// ─── Images ──────────────────────────────────────────────────────────────

export async function uploadImages(sessionName: string, files: File[]): Promise<{ uploaded: string[]; count: number }> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  const res = await fetch(`${API}/api/sessions/${encodeURIComponent(sessionName)}/images`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Failed to upload images");
  return res.json();
}

export function imageUrl(sessionName: string, filename: string): string {
  return `${API}/api/sessions/${encodeURIComponent(sessionName)}/images/${encodeURIComponent(filename)}`;
}

export async function deleteImage(sessionName: string, filename: string): Promise<void> {
  const res = await fetch(
    `${API}/api/sessions/${encodeURIComponent(sessionName)}/images/${encodeURIComponent(filename)}`,
    { method: "DELETE" }
  );
  if (!res.ok) throw new Error("Failed to delete image");
}

// ─── Video ───────────────────────────────────────────────────────────────

export async function extractVideoFrames(
  sessionName: string,
  file: File,
  frameInterval: number
): Promise<{ extracted: number; total_video_frames: number; fps: number }> {
  const form = new FormData();
  form.append("file", file);
  form.append("frame_interval", String(frameInterval));
  const res = await fetch(`${API}/api/sessions/${encodeURIComponent(sessionName)}/video`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Failed to extract frames");
  return res.json();
}

// ─── Annotations ─────────────────────────────────────────────────────────

export async function getAnnotation(sessionName: string, stem: string): Promise<AnnotationData | null> {
  const res = await fetch(
    `${API}/api/sessions/${encodeURIComponent(sessionName)}/annotations/${encodeURIComponent(stem)}`
  );
  if (!res.ok) return null;
  const data = await res.json();
  return data;
}

export async function saveAnnotation(
  sessionName: string,
  stem: string,
  data: AnnotationData
): Promise<void> {
  const res = await fetch(
    `${API}/api/sessions/${encodeURIComponent(sessionName)}/annotations/${encodeURIComponent(stem)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    }
  );
  if (!res.ok) throw new Error("Failed to save annotation");
}

export async function ignoreImage(sessionName: string, filename: string): Promise<void> {
  const res = await fetch(
    `${API}/api/sessions/${encodeURIComponent(sessionName)}/images/${encodeURIComponent(filename)}/ignore`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error("Failed to ignore image");
}

export async function getLastConditions(sessionName: string): Promise<Conditions | null> {
  const res = await fetch(
    `${API}/api/sessions/${encodeURIComponent(sessionName)}/last-conditions`
  );
  if (!res.ok) return null;
  const data = await res.json();
  return data;
}

// ─── Export ──────────────────────────────────────────────────────────────

export async function getExportStats(sessionName: string): Promise<ExportStats> {
  const res = await fetch(
    `${API}/api/sessions/${encodeURIComponent(sessionName)}/export/stats`
  );
  if (!res.ok) throw new Error("Failed to get export stats");
  return res.json();
}

export function exportDownloadUrl(sessionName: string): string {
  return `${API}/api/sessions/${encodeURIComponent(sessionName)}/export/download`;
}

// ─── Statistics ──────────────────────────────────────────────────────────

export async function getStatistics(sessionName: string): Promise<Statistics> {
  const res = await fetch(
    `${API}/api/sessions/${encodeURIComponent(sessionName)}/statistics`
  );
  if (!res.ok) throw new Error("Failed to get statistics");
  return res.json();
}
