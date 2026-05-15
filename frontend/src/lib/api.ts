const BASE = "http://localhost:8000";

export interface ImageMeta {
  filename: string;
  status: "to_annotate" | "annotated" | "ignored";
  added_at: string;
  source_video?: string;
}

export interface SessionMeta {
  name: string;
  description: string;
  folder: string;
  created_at: string;
  images: ImageMeta[];
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
  curvature_histogram: { range: string; count: number }[];
  angle_histogram: { range: string; count: number }[];
  camera_angles: { name: string; value: number }[];
  current_directions: { name: string; value: number }[];
  wave_scatter: { amplitude: number; speed: number }[];
  avg_points: number;
  annotators: { name: string; count: number }[];
  balance_warnings: string[];
}

export interface VideoProgress {
  current: number;
  total: number;
  status: "idle" | "running" | "done" | "error";
  error?: string;
}

export interface TrainProgress {
  epoch: number;
  total_epochs: number;
  status: "idle" | "starting" | "running" | "done" | "error";
  train_losses?: number[];
  val_losses?: number[];
  mae?: number;
  rmse?: number;
  preds?: number[];
  true?: number[];
  model_name?: string;
  n_train?: number;
  n_val?: number;
  device?: string;
  error?: string;
}

export interface ModelMeta {
  name: string;
  filename: string;
  size_mb: number;
  modified_at: string;
  is_global: boolean;
}

async function req(path: string, opts?: RequestInit) {
  const r = await fetch(`${BASE}${path}`, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  const ct = r.headers.get("content-type") || "";
  if (ct.includes("application/json")) return r.json();
  return r;
}

export const sanitizeName = async (name: string): Promise<string> => {
  const d = await req(`/api/sanitize?name=${encodeURIComponent(name)}`);
  return d.sanitized;
};

export const listSessions = (): Promise<SessionMeta[]> => req("/api/sessions");
export const createSession = (name: string, description = "", folder = "") => {
  const fd = new FormData();
  fd.append("name", name);
  fd.append("description", description);
  fd.append("folder", folder);
  return req("/api/sessions", { method: "POST", body: fd });
};
export const getSession = (name: string): Promise<SessionMeta> => req(`/api/sessions/${name}`);
export const deleteSession = (name: string) => req(`/api/sessions/${name}`, { method: "DELETE" });
export const batchMoveSessions = (names: string[], folder: string) =>
  req("/api/sessions/batch-move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ names, folder }),
  });

export const uploadImages = (name: string, files: File[]) => {
  const fd = new FormData();
  files.forEach(f => fd.append("files", f));
  return req(`/api/sessions/${name}/images`, { method: "POST", body: fd });
};
export const deleteImage = (name: string, filename: string) =>
  req(`/api/sessions/${name}/images/${filename}`, { method: "DELETE" });
export const imageUrl = (name: string, filename: string) =>
  `${BASE}/api/sessions/${name}/images/${filename}`;

export const extractVideoFrames = (name: string, file: File, frameInterval: number) => {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("frame_interval", String(frameInterval));
  return req(`/api/sessions/${name}/video`, { method: "POST", body: fd });
};
export const getVideoProgress = (name: string): Promise<VideoProgress> =>
  req(`/api/sessions/${name}/video/progress`);

export const getAnnotation = (name: string, stem: string) =>
  req(`/api/sessions/${name}/annotations/${stem}`);
export const saveAnnotation = (name: string, stem: string, data: object) =>
  req(`/api/sessions/${name}/annotations/${stem}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
export const ignoreImage = (name: string, filename: string) =>
  req(`/api/sessions/${name}/images/${filename}/ignore`, { method: "POST" });
export const getLastConditions = (name: string) =>
  req(`/api/sessions/${name}/last-conditions`);

export const getExportStats = (name: string): Promise<ExportStats> =>
  req(`/api/sessions/${name}/export/stats`);
export const exportDownloadUrl = (name: string) =>
  `${BASE}/api/sessions/${name}/export/download`;
export const getStatistics = (name: string): Promise<Statistics> =>
  req(`/api/sessions/${name}/statistics`);

export const getYoloStatus = () => req("/api/yolo/status");
export const predictAnnotation = (name: string, filename: string, conf = 0.5) =>
  req(`/api/sessions/${name}/images/${filename}/predict?conf=${conf}`);

export const trainSessionModel = (name: string, epochs = 50) => {
  const fd = new FormData();
  fd.append("epochs", String(epochs));
  return req(`/api/sessions/${name}/train`, { method: "POST", body: fd });
};
export const getTrainProgress = (name: string): Promise<TrainProgress> =>
  req(`/api/sessions/${name}/train/progress`);

export const trainGlobalModel = (opts: { sessions?: string; epochs?: number; model_name?: string }) => {
  const fd = new FormData();
  if (opts.sessions) fd.append("sessions", opts.sessions);
  fd.append("epochs", String(opts.epochs ?? 50));
  fd.append("model_name", opts.model_name ?? "global_vc_model");
  return req("/api/train/global", { method: "POST", body: fd });
};
export const getGlobalTrainProgress = (): Promise<TrainProgress> =>
  req("/api/train/global/progress");

export const listModels = (): Promise<ModelMeta[]> => req("/api/models");
export const predictVc = (modelName: string, file: File): Promise<{ vitesse_estimee: number }> => {
  const fd = new FormData();
  fd.append("model_name", modelName);
  fd.append("file", file);
  return req("/api/predict", { method: "POST", body: fd });
};
// Téléchargement direct d'un modèle
export const modelDownloadUrl = (filename: string) =>
  `${BASE}/api/models/${encodeURIComponent(filename)}/download`;
