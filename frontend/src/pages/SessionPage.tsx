import { useEffect, useState, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import {
  getSession, uploadImages, extractVideoFrames, deleteImage,
  getExportStats, exportDownloadUrl, getStatistics, imageUrl,
  type SessionMeta, type ExportStats, type Statistics,
} from "../lib/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell,
  ScatterChart, Scatter, CartesianGrid, ResponsiveContainer,
} from "recharts";

const CHART_COLORS = ["#00FFFF", "#00FF88", "#FFA500", "#FF4444", "#58A6FF", "#BC8CFF", "#FFD700", "#FF69B4"];

export default function SessionPage({ sessionName }: { sessionName: string }) {
  const [session, setSession] = useState<SessionMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"images" | "stats">("images");
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [frameInterval, setFrameInterval] = useState(240);
  const [extracting, setExtracting] = useState(false);
  const [extractResult, setExtractResult] = useState<string | null>(null);
  const [showExport, setShowExport] = useState(false);
  const [exportStats, setExportStats] = useState<ExportStats | null>(null);
  const [stats, setStats] = useState<Statistics | null>(null);
  const [, navigate] = useLocation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const data = await getSession(sessionName);
      setSession(data);
    } catch {
      navigate("/");
    } finally {
      setLoading(false);
    }
  }, [sessionName, navigate]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    if (tab === "stats") {
      getStatistics(sessionName).then(setStats).catch(() => {});
    }
  }, [tab, sessionName]);

  const handleFileDrop = async (files: FileList | File[]) => {
    const imgs = Array.from(files).filter((f) =>
      f.type.startsWith("image/") || /\.(jpg|jpeg|png)$/i.test(f.name)
    );
    if (imgs.length === 0) return;
    setUploading(true);
    try {
      await uploadImages(sessionName, imgs);
      await load();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setUploading(false);
    }
  };

  const handleVideoExtract = async () => {
    if (!videoFile) return;
    setExtracting(true);
    setExtractResult(null);
    try {
      const result = await extractVideoFrames(sessionName, videoFile, frameInterval);
      setExtractResult(`${result.extracted} frames extraites`);
      setVideoFile(null);
      await load();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setExtracting(false);
    }
  };

  const handleDeleteImage = async (filename: string) => {
    if (!confirm(`Supprimer ${filename} ?`)) return;
    try {
      await deleteImage(sessionName, filename);
      await load();
    } catch (e: any) {
      alert(e.message);
    }
  };

  const openExport = async () => {
    try {
      const s = await getExportStats(sessionName);
      setExportStats(s);
      setShowExport(true);
    } catch (e: any) {
      alert(e.message);
    }
  };

  const navigateAnnotate = () => {
    if (!session) return;
    navigate(`/session/${sessionName}/annotate`);
  };

  if (loading) {
    return <div style={{ textAlign: "center", padding: 60, color: "var(--color-text-dim)" }}>Chargement...</div>;
  }
  if (!session) return null;

  const totalImages = session.images.length;
  const annotatedCount = session.images.filter((i) => i.status === "annotated").length;
  const ignoredCount = session.images.filter((i) => i.status === "ignored").length;
  const pct = totalImages > 0 ? Math.round((annotatedCount / totalImages) * 100) : 0;

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
        <div>
          <button
            onClick={() => navigate("/")}
            style={{
              background: "none", border: "none", color: "var(--color-text-dim)",
              cursor: "pointer", fontSize: 13, marginBottom: 8, padding: 0,
            }}
          >
            ← Retour aux sessions
          </button>
          <h1 style={{ fontSize: 26, fontWeight: 700 }}>{session.name}</h1>
          {session.description && (
            <p style={{ color: "var(--color-text-dim)", fontSize: 14, marginTop: 4 }}>{session.description}</p>
          )}
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          {totalImages > 0 && (
            <button className="btn btn-green" onClick={navigateAnnotate}>
              🎯 Annoter
            </button>
          )}
          {annotatedCount > 0 && (
            <button className="btn btn-primary" onClick={openExport}>
              📦 Exporter dataset
            </button>
          )}
        </div>
      </div>

      {/* Progress */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <span style={{ fontSize: 14, fontWeight: 500 }}>Progression</span>
          <span style={{ fontSize: 14, color: "var(--color-accent)", fontWeight: 600 }}>{pct}%</span>
        </div>
        <div className="progress-bar" style={{ marginBottom: 12 }}>
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <span className="badge badge-cyan">{totalImages} images</span>
          <span className="badge badge-green">{annotatedCount} annotées</span>
          <span className="badge badge-orange">{ignoredCount} ignorées</span>
          <span className="badge badge-gray">{totalImages - annotatedCount - ignoredCount} restantes</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 24, borderBottom: "1px solid var(--color-border)", paddingBottom: 0 }}>
        {(["images", "stats"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              background: "none", border: "none", cursor: "pointer",
              padding: "10px 20px", fontSize: 14, fontWeight: 500,
              color: tab === t ? "var(--color-accent)" : "var(--color-text-dim)",
              borderBottom: tab === t ? "2px solid var(--color-accent)" : "2px solid transparent",
              transition: "all 0.2s",
            }}
          >
            {t === "images" ? "📷 Images" : "📊 Statistiques"}
          </button>
        ))}
      </div>

      {tab === "images" && (
        <div className="animate-fade-in">
          {/* Upload zones */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
            {/* Image upload */}
            <div
              className={`drop-zone ${dragOver ? "drag-over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFileDrop(e.dataTransfer.files); }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png"
                multiple
                hidden
                onChange={(e) => e.target.files && handleFileDrop(e.target.files)}
              />
              <div style={{ fontSize: 32, marginBottom: 8 }}>📷</div>
              <div style={{ fontWeight: 500, marginBottom: 4 }}>
                {uploading ? "Upload en cours..." : "Glissez-déposez des images"}
              </div>
              <div style={{ fontSize: 13, color: "var(--color-text-dim)" }}>JPG, PNG — Import multiple supporté</div>
            </div>

            {/* Video extraction */}
            <div className="card" style={{ padding: 24 }}>
              <div style={{ fontSize: 32, marginBottom: 8, textAlign: "center" }}>🎬</div>
              <div style={{ fontWeight: 500, marginBottom: 12, textAlign: "center" }}>Extraction vidéo</div>
              
              <div style={{ marginBottom: 12 }}>
                <button
                  className="btn btn-secondary"
                  style={{ width: "100%", justifyContent: "center" }}
                  onClick={() => videoInputRef.current?.click()}
                >
                  {videoFile ? videoFile.name : "Choisir une vidéo"}
                </button>
                <input
                  ref={videoInputRef}
                  type="file"
                  accept=".mp4,.avi,.mov"
                  hidden
                  onChange={(e) => e.target.files?.[0] && setVideoFile(e.target.files[0])}
                />
              </div>

              <div style={{ marginBottom: 12 }}>
                <label style={{ display: "block", fontSize: 12, color: "var(--color-text-dim)", marginBottom: 4 }}>
                  Intervalle de frames
                </label>
                <input
                  type="number"
                  value={frameInterval}
                  onChange={(e) => setFrameInterval(parseInt(e.target.value) || 1)}
                  min={1}
                  style={{ marginBottom: 4 }}
                />
                <div style={{ fontSize: 11, color: "var(--color-text-dim)" }}>
                  ex: 240 = 1 frame toutes les 8 secondes à 30fps
                </div>
              </div>

              <button
                className="btn btn-primary"
                style={{ width: "100%", justifyContent: "center" }}
                disabled={!videoFile || extracting}
                onClick={handleVideoExtract}
              >
                {extracting ? "Extraction en cours..." : "Extraire les frames"}
              </button>

              {extracting && (
                <div style={{ marginTop: 8 }}>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: "100%", animation: "pulse-glow 1.5s infinite" }} />
                  </div>
                </div>
              )}

              {extractResult && (
                <div style={{ marginTop: 8, fontSize: 13, color: "var(--color-green)", textAlign: "center", fontWeight: 500 }}>
                  ✓ {extractResult}
                </div>
              )}
            </div>
          </div>

          {/* Image grid */}
          {session.images.length > 0 && (
            <>
              <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>
                Images ({session.images.length})
              </h3>
              <div className="thumbnail-grid">
                {session.images.map((img, idx) => (
                  <div
                    key={img.filename}
                    className="thumbnail-card animate-fade-in"
                    style={{ animationDelay: `${Math.min(idx * 30, 300)}ms`, animationFillMode: "backwards" }}
                  >
                    <img
                      src={imageUrl(sessionName, img.filename)}
                      alt={img.filename}
                      loading="lazy"
                    />
                    <div
                      className="status-dot"
                      style={{
                        background:
                          img.status === "annotated" ? "var(--color-green)" :
                          img.status === "ignored" ? "var(--color-orange)" :
                          "var(--color-text-dim)",
                      }}
                    />
                    {/* Hover overlay */}
                    <div
                      style={{
                        position: "absolute", inset: 0, background: "rgba(0,0,0,0.7)",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        gap: 4, opacity: 0, transition: "opacity 0.2s",
                      }}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.opacity = "1"; }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.opacity = "0"; }}
                    >
                      <button
                        className="btn btn-red"
                        style={{ padding: "4px 8px", fontSize: 11 }}
                        onClick={() => handleDeleteImage(img.filename)}
                      >
                        🗑
                      </button>
                    </div>
                    <div
                      style={{
                        position: "absolute", bottom: 0, left: 0, right: 0,
                        background: "linear-gradient(transparent, rgba(0,0,0,0.8))",
                        padding: "16px 6px 4px",
                        fontSize: 10, color: "var(--color-text-dim)",
                        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                      }}
                    >
                      {img.filename}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {tab === "stats" && stats && (
        <StatsView stats={stats} />
      )}

      {/* Export modal */}
      {showExport && exportStats && (
        <div className="modal-backdrop" onClick={() => setShowExport(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: 20, fontWeight: 600, marginBottom: 16 }}>📦 Exporter le dataset</h2>
            
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
              <div className="card" style={{ textAlign: "center", padding: 16 }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: "var(--color-accent)" }}>{exportStats.total_annotated}</div>
                <div style={{ fontSize: 12, color: "var(--color-text-dim)" }}>Annotées</div>
              </div>
              <div className="card" style={{ textAlign: "center", padding: 16 }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: "var(--color-green)" }}>{exportStats.train_count}</div>
                <div style={{ fontSize: 12, color: "var(--color-text-dim)" }}>Train (80%)</div>
              </div>
              <div className="card" style={{ textAlign: "center", padding: 16 }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: "var(--color-orange)" }}>{exportStats.val_count}</div>
                <div style={{ fontSize: 12, color: "var(--color-text-dim)" }}>Val (20%)</div>
              </div>
            </div>

            <div style={{ color: "var(--color-text-dim)", marginBottom: 16, padding: 12, background: "var(--color-bg)", borderRadius: 8, fontFamily: "monospace", fontSize: 12 }}>
              dataset/<br />
              ├── images/train/ & val/<br />
              ├── labels/train/ & val/<br />
              ├── metadata/<br />
              ├── dataset.yaml<br />
              ├── dataset_summary.csv<br />
              └── train_yolo.py
            </div>

            <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setShowExport(false)}>Annuler</button>
              <a
                href={exportDownloadUrl(sessionName)}
                className="btn btn-green"
                style={{ textDecoration: "none" }}
                download
              >
                ⬇ Télécharger le ZIP
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Stats View ──────────────────────────────────────────────────────────────

function StatsView({ stats }: { stats: Statistics }) {
  return (
    <div className="animate-fade-in">
      {/* Progress */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 12 }}>📊 Avancement global</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: "var(--color-accent)" }}>{stats.total}</div>
            <div style={{ fontSize: 12, color: "var(--color-text-dim)" }}>Total</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: "var(--color-green)" }}>{stats.annotated}</div>
            <div style={{ fontSize: 12, color: "var(--color-text-dim)" }}>Annotées</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: "var(--color-orange)" }}>{stats.ignored}</div>
            <div style={{ fontSize: 12, color: "var(--color-text-dim)" }}>Ignorées</div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: 28, fontWeight: 700, color: "var(--color-text-dim)" }}>{stats.remaining}</div>
            <div style={{ fontSize: 12, color: "var(--color-text-dim)" }}>Restantes</div>
          </div>
        </div>
      </div>

      {/* Balance warnings */}
      {stats.balance_warnings.length > 0 && (
        <div
          className="card"
          style={{
            marginBottom: 24,
            borderColor: "rgba(255, 165, 0, 0.4)",
            background: "rgba(255, 165, 0, 0.05)",
          }}
        >
          <h3 style={{ fontSize: 14, fontWeight: 600, color: "var(--color-orange)", marginBottom: 8 }}>
            ⚠️ Alertes d'équilibre
          </h3>
          <div style={{ fontSize: 13, color: "var(--color-text-dim)" }}>
            {stats.balance_warnings.map((w, i) => (
              <div key={i} style={{ marginBottom: 4 }}>• {w} ({"<"}5 échantillons)</div>
            ))}
          </div>
        </div>
      )}

      {/* Charts grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
        {/* Speed histogram */}
        {stats.speed_histogram.length > 0 && (
          <div className="card">
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Vitesse courant (cm/s)</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={stats.speed_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="range" tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} />
                <YAxis tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} />
                <Tooltip
                  contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: "var(--color-text)" }}
                />
                <Bar dataKey="count" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Camera angles */}
        {stats.camera_angles.length > 0 && (
          <div className="card">
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Angles caméra</h3>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={stats.camera_angles} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name} (${value})`}>
                  {stats.camera_angles.map((_, i) => (
                    <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Current directions */}
        {stats.current_directions.length > 0 && (
          <div className="card">
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Direction courant</h3>
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={stats.current_directions} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name} (${value})`}>
                  {stats.current_directions.map((_, i) => (
                    <Cell key={i} fill={CHART_COLORS[(i + 3) % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Wave scatter */}
        {stats.wave_scatter.length > 0 && (
          <div className="card">
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 16 }}>Amplitude houle vs Vitesse courant</h3>
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="speed" name="Vitesse (cm/s)" tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} />
                <YAxis dataKey="amplitude" name="Amplitude (cm)" tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} />
                <Tooltip
                  contentStyle={{ background: "var(--color-surface)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }}
                  cursor={{ strokeDasharray: "3 3" }}
                />
                <Scatter data={stats.wave_scatter} fill="var(--color-green)" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Point stats + annotators */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div className="card">
          <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Points par annotation</h3>
          <div style={{ fontSize: 32, fontWeight: 700, color: "var(--color-accent)" }}>
            {stats.avg_points}
          </div>
          <div style={{ fontSize: 12, color: "var(--color-text-dim)" }}>points en moyenne</div>
        </div>

        {stats.annotators.length > 0 && (
          <div className="card">
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>Annotateurs</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {stats.annotators.map((a) => (
                <div key={a.name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 13 }}>{a.name}</span>
                  <span className="badge badge-cyan">{a.count} annotations</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
