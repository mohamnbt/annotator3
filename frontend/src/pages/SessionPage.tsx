import { useEffect, useState, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import {
  getSession, uploadImages, extractVideoFrames, deleteImage,
  getExportStats, exportDownloadUrl, getStatistics, imageUrl,
  getVideoProgress,
  type SessionMeta, type ExportStats, type Statistics, type VideoProgress,
} from "../lib/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, PieChart, Pie, Cell,
  ScatterChart, Scatter, CartesianGrid, ResponsiveContainer,
} from "recharts";
import AIPanel from "../components/AIPanel";

const C = {
  bg: "#0D1117", surface: "#161B22", surface2: "#21262D",
  accent: "#00FFFF", green: "#00FF88", orange: "#FFA500",
  red: "#FF4444", blue: "#3B82F6", border: "#30363D",
  text: "#E6EDF3", muted: "#8B949E",
};
const CHART_COLORS = [C.accent, C.green, C.orange, C.red, C.blue, "#BC8CFF", "#FFD700", "#FF69B4"];

type Tab = "images" | "annoter" | "stats" | "export" | "ia";

export default function SessionPage({ sessionName }: { sessionName: string }) {
  const [session, setSession] = useState<SessionMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<Tab>("images");
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [frameInterval, setFrameInterval] = useState(240);
  const [extracting, setExtracting] = useState(false);
  const [extractResult, setExtractResult] = useState<string | null>(null);
  const [videoProgress, setVideoProgress] = useState<VideoProgress | null>(null);
  const [showExport, setShowExport] = useState(false);
  const [exportStats, setExportStats] = useState<ExportStats | null>(null);
  const [stats, setStats] = useState<Statistics | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [, navigate] = useLocation();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

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
    if (tab === "stats") getStatistics(sessionName).then(setStats).catch(() => {});
  }, [tab, sessionName]);

  // Polling extraction vidéo
  const startVideoPoll = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const p = await getVideoProgress(sessionName);
        setVideoProgress(p);
        if (p.status === "done" || p.status === "error") {
          clearInterval(pollRef.current!);
          setExtracting(false);
          if (p.status === "done") {
            setExtractResult(`${p.current} frames extraites`);
            await load();
            showToast(`✓ ${p.current} frames extraites`);
          }
        }
      } catch {
        clearInterval(pollRef.current!);
        setExtracting(false);
      }
    }, 500);
  };

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const handleFileDrop = async (files: FileList | File[]) => {
    const imgs = Array.from(files).filter(
      (f) => f.type.startsWith("image/") || /\.(jpg|jpeg|png)$/i.test(f.name)
    );
    if (imgs.length === 0) return;
    setUploading(true);
    try {
      await uploadImages(sessionName, imgs);
      await load();
      showToast(`✓ ${imgs.length} image(s) importée(s)`);
    } catch (e: any) {
      showToast("Erreur upload : " + e.message);
    } finally {
      setUploading(false);
    }
  };

  const handleVideoExtract = async () => {
    if (!videoFile) return;
    setExtracting(true);
    setExtractResult(null);
    setVideoProgress({ current: 0, total: 0, status: "running" });
    try {
      await extractVideoFrames(sessionName, videoFile, frameInterval);
      setVideoFile(null);
      startVideoPoll();
    } catch (e: any) {
      showToast("Erreur extraction : " + e.message);
      setExtracting(false);
    }
  };

  const handleDeleteImage = async (filename: string) => {
    if (!confirm(`Supprimer ${filename} ?`)) return;
    try {
      await deleteImage(sessionName, filename);
      await load();
    } catch (e: any) {
      showToast("Erreur : " + e.message);
    }
  };

  const openExport = async () => {
    try {
      const s = await getExportStats(sessionName);
      setExportStats(s);
      setShowExport(true);
    } catch (e: any) {
      showToast("Erreur : " + e.message);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 80, color: C.muted }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>⚓</div>
        Chargement de la session...
      </div>
    );
  }
  if (!session) return null;

  const totalImages = session.images.length;
  const annotatedCount = session.images.filter((i) => i.status === "annotated").length;
  const ignoredCount = session.images.filter((i) => i.status === "ignored").length;
  const remainingCount = totalImages - annotatedCount - ignoredCount;
  const pct = totalImages > 0 ? Math.round((annotatedCount / totalImages) * 100) : 0;

  const tabs: { key: Tab; label: string; icon: string }[] = [
    { key: "images",  label: "Images",      icon: "📷" },
    { key: "annoter", label: "Annoter",      icon: "✏️" },
    { key: "stats",   label: "Statistiques", icon: "📊" },
    { key: "export",  label: "Exporter",     icon: "📦" },
    { key: "ia",      label: "IA / Modèle",  icon: "🧠" },
  ];

  return (
    <div className="animate-fade-in" style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px 40px" }}>

      {/* ── HEADER ── */}
      <div style={{ paddingTop: 28, paddingBottom: 20 }}>
        <button
          onClick={() => navigate("/")}
          style={{
            background: "none", border: "none", color: C.muted,
            cursor: "pointer", fontSize: 13, marginBottom: 12, padding: 0,
            display: "flex", alignItems: "center", gap: 6,
          }}
        >
          ← Retour aux sessions
        </button>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h1 style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.5px", marginBottom: 4 }}>
              {session.name}
            </h1>
            {session.description && (
              <p style={{ color: C.muted, fontSize: 14 }}>{session.description}</p>
            )}
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            {totalImages > 0 && (
              <button className="btn btn-green" onClick={() => navigate(`/session/${sessionName}/annotate`)}>
                ✏️ Annoter
              </button>
            )}
            {annotatedCount > 0 && (
              <button className="btn btn-primary" onClick={openExport}>
                📦 Exporter
              </button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        <div style={{ marginTop: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
            <div style={{ display: "flex", gap: 16 }}>
              <span style={{ color: C.accent, fontWeight: 600 }}>{annotatedCount} annotées</span>
              <span style={{ color: C.muted }}>{ignoredCount} ignorées</span>
              <span style={{ color: C.muted }}>{remainingCount} restantes</span>
            </div>
            <span style={{ color: C.accent, fontWeight: 700 }}>{pct}%</span>
          </div>
          <div style={{ height: 8, background: C.surface2, borderRadius: 4, overflow: "hidden" }}>
            <div style={{
              height: "100%", borderRadius: 4,
              width: `${pct}%`,
              background: pct === 100
                ? `linear-gradient(90deg, ${C.green}, ${C.accent})`
                : `linear-gradient(90deg, ${C.accent}, ${C.blue})`,
              transition: "width 0.5s ease",
            }} />
          </div>
          <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>{totalImages} images au total</div>
        </div>
      </div>

      {/* ── TABS ── */}
      <div style={{
        display: "flex", gap: 2, marginBottom: 28,
        borderBottom: `1px solid ${C.border}`,
      }}>
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{
              background: "none", border: "none", cursor: "pointer",
              padding: "10px 18px", fontSize: 13, fontWeight: 600,
              display: "flex", alignItems: "center", gap: 6,
              color: tab === t.key ? C.accent : C.muted,
              borderBottom: tab === t.key ? `2px solid ${C.accent}` : "2px solid transparent",
              transition: "all 0.18s",
              whiteSpace: "nowrap",
            }}
          >
            <span>{t.icon}</span> {t.label}
          </button>
        ))}
      </div>

      {/* ── ONGLET IMAGES ── */}
      {tab === "images" && (
        <div className="animate-fade-in">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 28 }}>
            {/* Upload images */}
            <div
              className={`drop-zone ${dragOver ? "drag-over" : ""}`}
              style={{ minHeight: 160 }}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFileDrop(e.dataTransfer.files); }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input ref={fileInputRef} type="file" accept="image/jpeg,image/png" multiple hidden
                onChange={(e) => e.target.files && handleFileDrop(e.target.files)} />
              <div style={{ fontSize: 36, marginBottom: 10 }}>📷</div>
              <div style={{ fontWeight: 600, marginBottom: 4, color: C.text }}>
                {uploading ? "Importation en cours..." : "Glissez-déposez des images"}
              </div>
              <div style={{ fontSize: 12, color: C.muted }}>JPG, PNG — Sélection multiple supportée</div>
            </div>

            {/* Extraction vidéo */}
            <div style={{
              background: C.surface, border: `1px solid ${C.border}`,
              borderRadius: 12, padding: 24,
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                <span style={{ fontSize: 24 }}>🎬</span>
                <span style={{ fontWeight: 700, fontSize: 15 }}>Extraction vidéo</span>
                <span style={{ fontSize: 10, color: C.muted, background: C.surface2, padding: "2px 8px", borderRadius: 10, border: `1px solid ${C.border}` }}>ffmpeg</span>
              </div>

              <div style={{ marginBottom: 12 }}>
                <button
                  className="btn btn-secondary"
                  style={{ width: "100%", justifyContent: "center" }}
                  onClick={() => videoInputRef.current?.click()}
                >
                  {videoFile ? `📹 ${videoFile.name}` : "Choisir une vidéo (MP4, AVI, MOV)"}
                </button>
                <input ref={videoInputRef} type="file" accept=".mp4,.avi,.mov" hidden
                  onChange={(e) => e.target.files?.[0] && setVideoFile(e.target.files[0])} />
              </div>

              <div style={{ marginBottom: 14 }}>
                <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 5, fontWeight: 500 }}>
                  Intervalle de frames
                </label>
                <input
                  type="number" value={frameInterval} min={1}
                  onChange={(e) => setFrameInterval(parseInt(e.target.value) || 1)}
                  style={{ width: 90, marginBottom: 4 }}
                />
                <div style={{ fontSize: 11, color: C.muted }}>ex: 240 = 1 frame toutes les 8s à 30fps</div>
              </div>

              <button
                className="btn btn-primary"
                style={{ width: "100%", justifyContent: "center" }}
                disabled={!videoFile || extracting}
                onClick={handleVideoExtract}
              >
                {extracting ? "⏳ Extraction en cours..." : "▶ Extraire les frames"}
              </button>

              {extracting && videoProgress && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: C.muted, marginBottom: 5 }}>
                    <span>Extraction...</span>
                    {videoProgress.total > 0 && (
                      <span>{videoProgress.current} / {videoProgress.total} frames</span>
                    )}
                  </div>
                  <div style={{ height: 6, background: C.border, borderRadius: 3, overflow: "hidden" }}>
                    <div style={{
                      height: "100%", borderRadius: 3,
                      background: `linear-gradient(90deg, ${C.accent}, ${C.blue})`,
                      width: videoProgress.total > 0
                        ? `${Math.round((videoProgress.current / videoProgress.total) * 100)}%`
                        : "100%",
                      transition: "width 0.4s",
                      animation: videoProgress.total === 0 ? "pulse-glow 1.5s infinite" : "none",
                    }} />
                  </div>
                </div>
              )}

              {extractResult && !extracting && (
                <div style={{ marginTop: 10, fontSize: 13, color: C.green, fontWeight: 600, textAlign: "center" }}>
                  ✓ {extractResult}
                </div>
              )}
            </div>
          </div>

          {/* Grille images */}
          {session.images.length > 0 && (
            <>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
                <h3 style={{ fontSize: 15, fontWeight: 700 }}>Images ({session.images.length})</h3>
                <div style={{ display: "flex", gap: 8 }}>
                  <span className="badge badge-green">{annotatedCount} ✓</span>
                  <span className="badge badge-orange">{ignoredCount} ignorées</span>
                  <span className="badge badge-gray">{remainingCount} restantes</span>
                </div>
              </div>
              <div className="thumbnail-grid">
                {session.images.map((img, idx) => (
                  <div key={img.filename} className="thumbnail-card animate-fade-in"
                    style={{ animationDelay: `${Math.min(idx * 25, 300)}ms`, animationFillMode: "backwards" }}>
                    <img src={imageUrl(sessionName, img.filename)} alt={img.filename} loading="lazy" />
                    <div className="status-dot" style={{
                      background: img.status === "annotated" ? C.green : img.status === "ignored" ? C.orange : C.muted,
                    }} />
                    <div style={{
                      position: "absolute", inset: 0, background: "rgba(0,0,0,0.75)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      gap: 6, opacity: 0, transition: "opacity 0.2s",
                    }}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.opacity = "1"; }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.opacity = "0"; }}
                    >
                      <button className="btn btn-red" style={{ padding: "4px 8px", fontSize: 11 }}
                        onClick={() => handleDeleteImage(img.filename)}>🗑</button>
                    </div>
                    <div style={{
                      position: "absolute", bottom: 0, left: 0, right: 0,
                      background: "linear-gradient(transparent, rgba(0,0,0,0.85))",
                      padding: "16px 6px 4px", fontSize: 9, color: "rgba(255,255,255,0.6)",
                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                    }}>{img.filename}</div>
                  </div>
                ))}
              </div>
            </>
          )}

          {session.images.length === 0 && (
            <div style={{ textAlign: "center", padding: "48px 0", color: C.muted }}>
              <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.4 }}>🖼</div>
              Aucune image pour le moment — importez des images ou extrayez des frames vidéo.
            </div>
          )}
        </div>
      )}

      {/* ── ONGLET ANNOTER ── */}
      {tab === "annoter" && (
        <div className="animate-fade-in" style={{ textAlign: "center", padding: "48px 0" }}>
          <div style={{ fontSize: 56, marginBottom: 20 }}>✏️</div>
          <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 10 }}>Interface d'annotation</h2>
          <p style={{ color: C.muted, marginBottom: 28, maxWidth: 400, margin: "0 auto 28px" }}>
            Tracez la ligne centrale du câble, renseignez les conditions expérimentales
            (notamment <span style={{ color: C.accent }}>Vitesse courant (cm/s)</span>) puis confirmez.
          </p>
          {totalImages === 0 ? (
            <p style={{ color: C.orange }}>⚠️ Importez d'abord des images dans l'onglet Images.</p>
          ) : (
            <button
              className="btn btn-green"
              style={{ fontSize: 15, padding: "12px 32px" }}
              onClick={() => navigate(`/session/${sessionName}/annotate`)}
            >
              🚀 Ouvrir l'annotateur ({remainingCount} restantes)
            </button>
          )}
          <div style={{ marginTop: 36, display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, maxWidth: 560, margin: "36px auto 0" }}>
            {[
              { icon: "🖱️", title: "Clic gauche", desc: "Ajoute un point" },
              { icon: "⌨️", title: "Entrée", desc: "Confirme l'annotation" },
              { icon: "🔍", title: "Molette", desc: "Zoom centré curseur" },
            ].map((tip) => (
              <div key={tip.title} style={{
                background: C.surface, border: `1px solid ${C.border}`,
                borderRadius: 10, padding: 16,
              }}>
                <div style={{ fontSize: 24, marginBottom: 8 }}>{tip.icon}</div>
                <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{tip.title}</div>
                <div style={{ fontSize: 12, color: C.muted }}>{tip.desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── ONGLET STATISTIQUES ── */}
      {tab === "stats" && (
        <div className="animate-fade-in">
          {!stats ? (
            <div style={{ textAlign: "center", padding: 60, color: C.muted }}>Chargement des statistiques...</div>
          ) : (
            <StatsView stats={stats} />
          )}
        </div>
      )}

      {/* ── ONGLET EXPORT ── */}
      {tab === "export" && (
        <div className="animate-fade-in">
          <div style={{
            background: C.surface, border: `1px solid ${C.border}`,
            borderRadius: 14, padding: 28, maxWidth: 580,
          }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>📦 Exporter le dataset YOLO</h2>
            <p style={{ color: C.muted, fontSize: 14, marginBottom: 20 }}>
              Génère un ZIP avec images, labels YOLO, metadata JSON,
              dataset.yaml et un script train_yolo.py prêt à l'emploi.
            </p>
            {annotatedCount === 0 ? (
              <div style={{ color: C.orange, fontSize: 14 }}>
                ⚠️ Aucune image annotée. Annotez des images avant d'exporter.
              </div>
            ) : (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
                  <StatCard v={annotatedCount} label="Annotées" color={C.accent} />
                  <StatCard v={Math.floor(annotatedCount * 0.8)} label="Train (80%)" color={C.green} />
                  <StatCard v={annotatedCount - Math.floor(annotatedCount * 0.8)} label="Val (20%)" color={C.orange} />
                </div>
                <div style={{
                  background: C.bg, border: `1px solid ${C.border}`, borderRadius: 8,
                  padding: "12px 16px", fontFamily: "monospace", fontSize: 12, color: C.muted, marginBottom: 20,
                }}>
                  dataset/<br />
                  ├── images/train/ &amp; val/<br />
                  ├── labels/train/ &amp; val/<br />
                  ├── metadata/ (.json par image)<br />
                  ├── dataset.yaml<br />
                  ├── dataset_summary.csv<br />
                  └── train_yolo.py
                </div>
                <a
                  href={exportDownloadUrl(sessionName)}
                  className="btn btn-green"
                  style={{ textDecoration: "none", justifyContent: "center", display: "inline-flex", padding: "10px 24px", fontSize: 14 }}
                  download
                >
                  ⬇ Télécharger le dataset.zip
                </a>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── ONGLET IA ── */}
      {tab === "ia" && (
        <div className="animate-fade-in">
          <AIPanel sessionName={sessionName} />
        </div>
      )}

      {/* ── MODAL EXPORT LEGACY ── */}
      {showExport && exportStats && (
        <div className="modal-backdrop" onClick={() => setShowExport(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>📦 Exporter le dataset</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
              <StatCard v={exportStats.total_annotated} label="Annotées" color={C.accent} />
              <StatCard v={exportStats.train_count} label="Train (80%)" color={C.green} />
              <StatCard v={exportStats.val_count} label="Val (20%)" color={C.orange} />
            </div>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setShowExport(false)}>Annuler</button>
              <a href={exportDownloadUrl(sessionName)} className="btn btn-green" style={{ textDecoration: "none" }} download>
                ⬇ Télécharger
              </a>
            </div>
          </div>
        </div>
      )}

      {/* Toast */}
      {toast && (
        <div style={{
          position: "fixed", bottom: 24, right: 24, zIndex: 9999,
          background: C.surface, border: `1px solid ${C.border}`,
          padding: "12px 20px", borderRadius: 10,
          fontSize: 14, color: C.green, fontWeight: 600,
          boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
          animation: "fade-in 0.2s ease",
        }}>
          {toast}
        </div>
      )}
    </div>
  );
}

// ── StatCard ──────────────────────────────────────────────────────────────────
function StatCard({ v, label, color }: { v: number; label: string; color: string }) {
  return (
    <div style={{
      background: "#0D1117", borderRadius: 10, padding: "14px 12px",
      border: "1px solid #30363D", textAlign: "center",
    }}>
      <div style={{ fontSize: 26, fontWeight: 800, color }}>{v}</div>
      <div style={{ fontSize: 12, color: "#8B949E", marginTop: 2 }}>{label}</div>
    </div>
  );
}

// ── StatsView ─────────────────────────────────────────────────────────────────
function StatsView({ stats }: { stats: Statistics }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Counters */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        {[
          { v: stats.total,     label: "Total",     color: C.accent },
          { v: stats.annotated, label: "Annotées",  color: C.green },
          { v: stats.ignored,   label: "Ignorées",  color: C.orange },
          { v: stats.remaining, label: "Restantes", color: C.muted },
        ].map((s) => (
          <div key={s.label} style={{
            background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12,
            padding: "18px 16px", textAlign: "center",
          }}>
            <div style={{ fontSize: 30, fontWeight: 800, color: s.color }}>{s.v}</div>
            <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Balance warnings */}
      {stats.balance_warnings.length > 0 && (
        <div style={{
          background: "rgba(255,165,0,0.06)", border: "1px solid rgba(255,165,0,0.3)",
          borderRadius: 10, padding: "14px 18px",
        }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: C.orange, marginBottom: 8 }}>⚠️ Alertes d'équilibre</div>
          {stats.balance_warnings.map((w, i) => (
            <div key={i} style={{ fontSize: 13, color: C.muted, marginBottom: 3 }}>• {w} (&lt;5 échantillons)</div>
          ))}
        </div>
      )}

      {/* Charts grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {stats.speed_histogram.length > 0 && (
          <ChartCard title="Distribution Vc (cm/s)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={stats.speed_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="range" tick={{ fontSize: 10, fill: C.muted }} />
                <YAxis tick={{ fontSize: 10, fill: C.muted }} />
                <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="count" fill={C.accent} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
        {stats.camera_angles.length > 0 && (
          <ChartCard title="Angles caméra">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={stats.camera_angles} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false} fontSize={10}>
                  {stats.camera_angles.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
        {stats.current_directions.length > 0 && (
          <ChartCard title="Types de courant">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={stats.current_directions} dataKey="value" nameKey="name" cx="50%" cy="50%"
                  outerRadius={80} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={false} fontSize={10}>
                  {stats.current_directions.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
        {stats.wave_scatter.length > 0 && (
          <ChartCard title="Houle vs Vc">
            <ResponsiveContainer width="100%" height={220}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="amplitude" name="Amplitude (cm)" tick={{ fontSize: 10, fill: C.muted }} />
                <YAxis dataKey="speed" name="Vc (cm/s)" tick={{ fontSize: 10, fill: C.muted }} />
                <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, fontSize: 12 }} />
                <Scatter data={stats.wave_scatter} fill={C.green} opacity={0.8} />
              </ScatterChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
        {stats.angle_histogram.length > 0 && (
          <ChartCard title="Angle câble θ (°)">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={stats.angle_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="range" tick={{ fontSize: 10, fill: C.muted }} />
                <YAxis tick={{ fontSize: 10, fill: C.muted }} />
                <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="count" fill={C.orange} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
        {stats.curvature_histogram.length > 0 && (
          <ChartCard title="Indice de courbure">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={stats.curvature_histogram}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="range" tick={{ fontSize: 10, fill: C.muted }} />
                <YAxis tick={{ fontSize: 10, fill: C.muted }} />
                <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="count" fill={C.blue} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        )}
      </div>

      {/* Annotateurs */}
      {stats.annotators.length > 0 && (
        <div style={{
          background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, overflow: "hidden",
        }}>
          <div style={{ padding: "14px 18px", borderBottom: `1px solid ${C.border}`, fontWeight: 700, fontSize: 14 }}>
            👤 Annotateurs
          </div>
          <div style={{ padding: "8px 0" }}>
            {stats.annotators.map((a) => (
              <div key={a.name} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "8px 18px",
              }}>
                <span style={{ fontSize: 14 }}>{a.name}</span>
                <span style={{
                  background: C.accent + "22", color: C.accent,
                  padding: "2px 12px", borderRadius: 20, fontSize: 13, fontWeight: 700,
                }}>{a.count}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`,
      borderRadius: 12, overflow: "hidden",
    }}>
      <div style={{ padding: "12px 16px", borderBottom: `1px solid ${C.border}`, fontSize: 13, fontWeight: 700 }}>
        {title}
      </div>
      <div style={{ padding: 16 }}>{children}</div>
    </div>
  );
}
