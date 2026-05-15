import { useEffect, useState, useCallback, useRef } from "react";
import { useLocation } from "wouter";
import {
  listSessions, createSession, deleteSession, sanitizeName,
  batchMoveSessions, trainGlobalModel, getGlobalTrainProgress,
  listModels, predictVc,
  type SessionMeta, type ModelMeta, type TrainProgress,
} from "../lib/api";

const C = {
  bg: "#0D1117", surface: "#161B22", surface2: "#21262D",
  accent: "#00FFFF", green: "#00FF88", orange: "#FFA500",
  red: "#FF4444", blue: "#3B82F6", border: "#30363D",
  text: "#E6EDF3", muted: "#8B949E",
};

type ActiveTab = "sessions" | "ai";

export default function HomePage() {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<ActiveTab>("sessions");

  // ─── Dossiers ────────────────────────────────────────────────────────────
  const [folders, setFolders] = useState<Record<string, SessionMeta[]>>({});
  const [openFolders, setOpenFolders] = useState<Record<string, boolean>>({});
  const [showMoveModal, setShowMoveModal] = useState(false);
  const [moveFolderName, setMoveFolderName] = useState("");
  const [showCreateFolder, setShowCreateFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");

  // ─── Sélection ───────────────────────────────────────────────────────────
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [isDragging, setIsDragging] = useState(false);
  const [dragRect, setDragRect] = useState<{x:number;y:number;w:number;h:number} | null>(null);
  const dragStart = useRef<{x:number;y:number} | null>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});

  // ─── Session create/delete ───────────────────────────────────────────────
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [deleteSelected, setDeleteSelected] = useState(false);

  // ─── AI Global tab ───────────────────────────────────────────────────────
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [trainSessions, setTrainSessions] = useState<Set<string>>(new Set());
  const [trainFolderTarget, setTrainFolderTarget] = useState<string>("");
  const [globalModelName, setGlobalModelName] = useState("global_vc_model");
  const [globalEpochs, setGlobalEpochs] = useState(50);
  const [globalProgress, setGlobalProgress] = useState<TrainProgress | null>(null);
  const [globalTrainResult, setGlobalTrainResult] = useState<TrainProgress | null>(null);
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predictModel, setPredictModel] = useState("");
  const [predictResult, setPredictResult] = useState<number | null>(null);
  const [predictLoading, setPredictLoading] = useState(false);

  const [, navigate] = useLocation();

  const load = useCallback(async () => {
    try {
      const data = await listSessions();
      setSessions(data);
      // Organiser par dossier
      const grouped: Record<string, SessionMeta[]> = { "": [] };
      for (const s of data) {
        const f = s.folder || "";
        if (!grouped[f]) grouped[f] = [];
        grouped[f].push(s);
      }
      setFolders(grouped);
    } catch {
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadModels = useCallback(async () => {
    try { setModels(await listModels()); } catch { setModels([]); }
  }, []);

  useEffect(() => { load(); loadModels(); }, [load, loadModels]);

  // ─── Polling entraînement global ─────────────────────────────────────────
  useEffect(() => {
    if (!globalProgress || globalProgress.status === "done" || globalProgress.status === "error" || globalProgress.status === "idle") return;
    const t = setInterval(async () => {
      try {
        const p = await getGlobalTrainProgress();
        setGlobalProgress(p);
        if (p.status === "done") { setGlobalTrainResult(p); clearInterval(t); loadModels(); }
        if (p.status === "error") { clearInterval(t); }
      } catch { clearInterval(t); }
    }, 500);
    return () => clearInterval(t);
  }, [globalProgress?.status]);

  // ─── Sélection lasso ─────────────────────────────────────────────────────
  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest("button") || (e.target as HTMLElement).closest("[data-card]")) return;
    if (e.button !== 0) return;
    dragStart.current = { x: e.clientX, y: e.clientY };
    setIsDragging(true);
    setDragRect({ x: e.clientX, y: e.clientY, w: 0, h: 0 });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !dragStart.current) return;
    const x = Math.min(e.clientX, dragStart.current.x);
    const y = Math.min(e.clientY, dragStart.current.y);
    const w = Math.abs(e.clientX - dragStart.current.x);
    const h = Math.abs(e.clientY - dragStart.current.y);
    setDragRect({ x, y, w, h });
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    if (!isDragging || !dragStart.current || !dragRect) { setIsDragging(false); setDragRect(null); return; }
    const lassoRect = {
      left: Math.min(e.clientX, dragStart.current.x),
      top: Math.min(e.clientY, dragStart.current.y),
      right: Math.max(e.clientX, dragStart.current.x),
      bottom: Math.max(e.clientY, dragStart.current.y),
    };
    if (lassoRect.right - lassoRect.left > 10 || lassoRect.bottom - lassoRect.top > 10) {
      const newSel = new Set(selected);
      for (const [name, el] of Object.entries(cardRefs.current)) {
        if (!el) continue;
        const r = el.getBoundingClientRect();
        const overlap = r.left < lassoRect.right && r.right > lassoRect.left && r.top < lassoRect.bottom && r.bottom > lassoRect.top;
        if (overlap) {
          if (newSel.has(name)) newSel.delete(name);
          else newSel.add(name);
        }
      }
      setSelected(newSel);
    }
    setIsDragging(false);
    setDragRect(null);
    dragStart.current = null;
  };

  const toggleSelect = (name: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const s = new Set(selected);
    if (s.has(name)) s.delete(name); else s.add(name);
    setSelected(s);
  };

  // ─── Dossiers ────────────────────────────────────────────────────────────
  const toggleFolder = (name: string) => setOpenFolders(p => ({ ...p, [name]: !p[name] }));

  const handleMoveToFolder = async () => {
    if (!selected.size) return;
    await batchMoveSessions(Array.from(selected), moveFolderName);
    setSelected(new Set());
    setShowMoveModal(false);
    setMoveFolderName("");
    await load();
  };

  // ─── Entraînement depuis dossier ─────────────────────────────────────────
  const allFolderNames = Object.keys(folders).filter(f => f !== "");

  const handleTrainFromFolder = async () => {
    const sessionsInFolder = folders[trainFolderTarget] || [];
    const names = sessionsInFolder.map(s => s.name);
    if (!names.length) return;
    setTrainSessions(new Set(names));
    await handleLaunchGlobalTrain(names);
  };

  const handleLaunchGlobalTrain = async (forceNames?: string[]) => {
    const names = forceNames || Array.from(trainSessions);
    const sessStr = names.join(",");
    setGlobalTrainResult(null);
    try {
      await trainGlobalModel({ sessions: sessStr, epochs: globalEpochs, model_name: globalModelName });
      setGlobalProgress({ epoch: 0, total_epochs: globalEpochs, status: "starting" });
    } catch (e: any) {
      alert(e.message);
    }
  };

  const handleNameChange = async (val: string) => {
    setNewName(val);
    if (val.length > 2) {
      const s = await sanitizeName(val).catch(() => val);
      if (s !== val) setNewName(s);
    }
  };

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      await createSession(newName, newDesc);
      setShowCreate(false); setNewName(""); setNewDesc("");
      await load();
    } catch (e: any) { alert(e.message); } finally { setCreating(false); }
  };

  const handleDeleteSelected = async () => {
    for (const name of selected) {
      try { await deleteSession(name); } catch {}
    }
    setSelected(new Set());
    setDeleteSelected(false);
    await load();
  };

  const handleDelete = async (name: string) => {
    try { await deleteSession(name); await load(); } catch (e: any) { alert(e.message); } finally { setDeleteTarget(null); }
  };

  const handlePredict = async () => {
    if (!predictFile || !predictModel) return;
    setPredictLoading(true); setPredictResult(null);
    try {
      const r = await predictVc(predictModel, predictFile);
      setPredictResult(r.vitesse_estimee);
    } catch (e: any) { alert(e.message); } finally { setPredictLoading(false); }
  };

  const totalAnnotated = sessions.reduce((acc, s) => acc + s.images.filter(i => i.status === "annotated").length, 0);
  const noFolder = folders[""] || [];
  const folderNames = Object.keys(folders).filter(f => f !== "");

  // ─── Render card ─────────────────────────────────────────────────────────
  const renderCard = (s: SessionMeta) => {
    const ann = s.images.filter(i => i.status === "annotated").length;
    const ign = s.images.filter(i => i.status === "ignored").length;
    const total = s.images.length;
    const pct = total > 0 ? Math.round((ann / total) * 100) : 0;
    const isSelected = selected.has(s.name);
    return (
      <div
        key={s.name}
        data-card="true"
        ref={el => { cardRefs.current[s.name] = el; }}
        style={{
          background: isSelected ? `${C.surface2}` : C.surface,
          border: `2px solid ${isSelected ? C.accent : C.border}`,
          borderRadius: 14, padding: 20, cursor: "pointer",
          transition: "border-color 0.15s, transform 0.15s, box-shadow 0.15s",
          boxShadow: isSelected ? `0 0 0 2px ${C.accent}33` : "none",
          userSelect: "none",
        }}
        onMouseEnter={e => {
          if (!isSelected) (e.currentTarget as HTMLElement).style.borderColor = C.accent + "60";
          (e.currentTarget as HTMLElement).style.transform = "translateY(-2px)";
        }}
        onMouseLeave={e => {
          if (!isSelected) (e.currentTarget as HTMLElement).style.borderColor = C.border;
          (e.currentTarget as HTMLElement).style.transform = "";
        }}
        onClick={e => {
          if (e.shiftKey || e.ctrlKey || e.metaKey) { toggleSelect(s.name, e); return; }
          navigate(`/session/${s.name}`);
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flex: 1, minWidth: 0 }}>
            {/* Checkbox sélection */}
            <div
              onClick={e => toggleSelect(s.name, e)}
              style={{
                width: 18, height: 18, borderRadius: 4, border: `2px solid ${isSelected ? C.accent : C.muted}`,
                background: isSelected ? C.accent : "transparent",
                flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center",
                transition: "all 0.15s", cursor: "pointer",
              }}
            >
              {isSelected && <span style={{ fontSize: 10, color: C.bg, fontWeight: 900 }}>✓</span>}
            </div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{s.name}</div>
              {s.description && (
                <div style={{ fontSize: 12, color: C.muted, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {s.description}
                </div>
              )}
            </div>
          </div>
          <button
            className="btn btn-red"
            style={{ padding: "3px 8px", fontSize: 12, flexShrink: 0, marginLeft: 8 }}
            onClick={e => { e.stopPropagation(); setDeleteTarget(s.name); }}
          >🗑</button>
        </div>
        <div style={{ marginBottom: 10 }}>
          <div style={{ height: 5, background: C.surface2, borderRadius: 3, overflow: "hidden", marginBottom: 5 }}>
            <div style={{
              height: "100%", borderRadius: 3, width: `${pct}%`,
              background: pct === 100 ? `linear-gradient(90deg,${C.green},${C.accent})` : `linear-gradient(90deg,${C.accent},${C.blue})`,
              transition: "width 0.4s",
            }} />
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: C.muted }}>
            <span>{ann} / {total}</span>
            <span style={{ color: pct === 100 ? C.green : C.accent, fontWeight: 700 }}>{pct}%</span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
          {ann > 0 && <span className="badge badge-green">{ann} annotées</span>}
          {ign > 0 && <span className="badge badge-orange">{ign} ign.</span>}
          {total - ann - ign > 0 && <span className="badge badge-gray">{total - ann - ign} à faire</span>}
          {total === 0 && <span className="badge badge-gray">Vide</span>}
        </div>
        <div style={{ fontSize: 11, color: C.muted, marginTop: 10 }}>
          {new Date(s.created_at).toLocaleDateString("fr-FR")}
        </div>
      </div>
    );
  };

  return (
    <div
      className="animate-fade-in"
      style={{ maxWidth: 1140, margin: "0 auto", padding: "0 24px 60px", position: "relative" }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* ── Lasso rect ── */}
      {isDragging && dragRect && dragRect.w > 5 && (
        <div style={{
          position: "fixed",
          left: dragRect.x, top: dragRect.y,
          width: dragRect.w, height: dragRect.h,
          border: `1.5px dashed ${C.accent}`,
          background: `${C.accent}15`,
          pointerEvents: "none", zIndex: 9999, borderRadius: 4,
        }} />
      )}

      {/* ── HERO ── */}
      <div style={{ paddingTop: 36, paddingBottom: 28, borderBottom: `1px solid ${C.border}`, marginBottom: 0 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
              <span style={{ fontSize: 36 }}>🤿</span>
              <div>
                <h1 style={{
                  fontSize: 28, fontWeight: 900, letterSpacing: "-0.8px",
                  background: `linear-gradient(90deg,${C.accent},${C.blue})`,
                  WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                }}>COSMER Annotator</h1>
                <div style={{ fontSize: 12, color: C.muted }}>Laboratoire COSMER — Université de Toulon</div>
              </div>
            </div>
            <p style={{ color: C.muted, fontSize: 13, maxWidth: 500, lineHeight: 1.6 }}>
              Annotez des câbles de mouillage immergés pour estimer la{" "}
              <span style={{ color: C.accent, fontWeight: 600 }}>vitesse du courant marin Vc</span>{" "}
              par deep learning.
            </p>
          </div>
          <div style={{ display: "flex", gap: 8, flexShrink: 0, marginTop: 4 }}>
            {totalAnnotated > 0 && (
              <a href="http://localhost:8000/api/export/global/download" className="btn btn-secondary" style={{ textDecoration: "none" }} download>
                🌍 Export global
              </a>
            )}
            <button className="btn btn-primary" onClick={() => setShowCreate(true)}>+ Nouvelle session</button>
          </div>
        </div>
        {sessions.length > 0 && (
          <div style={{ display: "flex", gap: 24, marginTop: 20 }}>
            {[
              { v: sessions.length, label: "Sessions", color: C.accent },
              { v: sessions.reduce((a, s) => a + s.images.length, 0), label: "Images", color: C.text },
              { v: totalAnnotated, label: "Annotées", color: C.green },
              { v: folderNames.length, label: "Dossiers", color: C.orange },
            ].map(s => (
              <div key={s.label} style={{ fontSize: 13, color: C.muted }}>
                <span style={{ fontSize: 20, fontWeight: 800, color: s.color, marginRight: 5 }}>{s.v}</span>{s.label}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── TABS ── */}
      <div style={{ display: "flex", gap: 0, borderBottom: `1px solid ${C.border}`, marginBottom: 28 }}>
        {(["sessions", "ai"] as ActiveTab[]).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: "12px 20px", fontSize: 13, fontWeight: 600,
              background: "transparent", border: "none",
              borderBottom: `2px solid ${activeTab === tab ? C.accent : "transparent"}`,
              color: activeTab === tab ? C.accent : C.muted,
              cursor: "pointer", transition: "color 0.15s",
            }}
          >
            {tab === "sessions" ? "📂 Sessions" : "🧠 IA Globale"}
          </button>
        ))}
      </div>

      {/* ═══ TAB SESSIONS ═══ */}
      {activeTab === "sessions" && (
        <>
          {/* ── Toolbar sélection ── */}
          {selected.size > 0 && (
            <div style={{
              display: "flex", alignItems: "center", gap: 10,
              background: C.surface2, border: `1px solid ${C.accent}40`,
              borderRadius: 10, padding: "10px 16px", marginBottom: 20,
            }}>
              <span style={{ fontSize: 13, color: C.accent, fontWeight: 700 }}>
                {selected.size} sélectionnée{selected.size > 1 ? "s" : ""}
              </span>
              <button className="btn btn-secondary" style={{ padding: "4px 12px", fontSize: 12 }}
                onClick={() => setShowMoveModal(true)}>
                📁 Déplacer vers dossier
              </button>
              <button className="btn btn-red" style={{ padding: "4px 12px", fontSize: 12 }}
                onClick={() => setDeleteSelected(true)}>
                🗑 Supprimer la sélection
              </button>
              <button className="btn btn-secondary" style={{ padding: "4px 12px", fontSize: 12, marginLeft: "auto" }}
                onClick={() => setSelected(new Set())}>
                ✕ Désélectionner
              </button>
            </div>
          )}

          {/* ── Bouton nouveau dossier ── */}
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16 }}>
            <button className="btn btn-secondary" style={{ fontSize: 12 }} onClick={() => setShowCreateFolder(true)}>
              + Nouveau dossier
            </button>
          </div>

          {loading ? (
            <div style={{ textAlign: "center", padding: 60, color: C.muted }}>
              <div style={{ fontSize: 32, marginBottom: 12 }}>⚓</div>Chargement...
            </div>
          ) : sessions.length === 0 ? (
            <div style={{ textAlign: "center", padding: "60px 0" }}>
              <div style={{ fontSize: 52, marginBottom: 16, opacity: 0.3 }}>📂</div>
              <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>Aucune session</h2>
              <p style={{ color: C.muted, marginBottom: 24 }}>Créez votre première session pour commencer.</p>
              <button className="btn btn-primary" onClick={() => setShowCreate(true)}>+ Créer une session</button>
            </div>
          ) : (
            <>
              {/* ── Dossiers nommés ── */}
              {folderNames.sort().map(folderName => (
                <div key={folderName} style={{ marginBottom: 24 }}>
                  <button
                    style={{
                      display: "flex", alignItems: "center", gap: 8,
                      background: "none", border: "none", cursor: "pointer",
                      color: C.orange, fontSize: 14, fontWeight: 700, padding: "6px 0", marginBottom: 12,
                    }}
                    onClick={() => toggleFolder(folderName)}
                  >
                    <span style={{ transition: "transform 0.2s", display: "inline-block", transform: openFolders[folderName] === false ? "rotate(-90deg)" : "" }}>▼</span>
                    📁 {folderName}
                    <span style={{ fontSize: 12, color: C.muted, fontWeight: 400 }}>({folders[folderName]?.length ?? 0} sessions)</span>
                  </button>
                  {openFolders[folderName] !== false && (
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(300px,1fr))", gap: 14, paddingLeft: 12, borderLeft: `2px solid ${C.orange}40` }}>
                      {(folders[folderName] || []).map(renderCard)}
                    </div>
                  )}
                </div>
              ))}
              {/* ── Sans dossier ── */}
              {noFolder.length > 0 && (
                <div style={{ marginBottom: 24 }}>
                  {folderNames.length > 0 && (
                    <div style={{ fontSize: 13, color: C.muted, fontWeight: 600, marginBottom: 12 }}>📋 Sans dossier</div>
                  )}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(300px,1fr))", gap: 14 }}>
                    {noFolder.map(renderCard)}
                  </div>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ═══ TAB IA GLOBALE ═══ */}
      {activeTab === "ai" && (
        <div style={{ maxWidth: 820, margin: "0 auto" }}>
          {/* Hero IA */}
          <div style={{
            background: `linear-gradient(135deg, #0D2137 0%, #0D1117 100%)`,
            border: `1px solid ${C.accent}30`, borderRadius: 16, padding: "28px 32px", marginBottom: 32,
            textAlign: "center",
          }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>🌊</div>
            <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 8, color: C.accent }}>Entraînement IA Global</h2>
            <p style={{ color: C.muted, fontSize: 13, lineHeight: 1.7, maxWidth: 520, margin: "0 auto" }}>
              Entraîne un <strong style={{ color: C.text }}>ResNet18</strong> sur plusieurs sessions à la fois.
              Idéal pour entraîner depuis un <strong style={{ color: C.orange }}>dossier complet</strong> de sessions.
            </p>
          </div>

          {/* ─ Entraîner depuis dossier ─ */}
          {allFolderNames.length > 0 && (
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 14, padding: 24, marginBottom: 24 }}>
              <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 16 }}>📁 Entraîner depuis un dossier</div>
              <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <select
                  value={trainFolderTarget}
                  onChange={e => setTrainFolderTarget(e.target.value)}
                  style={{ flex: 1, minWidth: 180, padding: "8px 12px", background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, fontSize: 13 }}
                >
                  <option value="">— Choisir un dossier —</option>
                  {allFolderNames.map(f => (
                    <option key={f} value={f}>{f} ({folders[f]?.length ?? 0} sessions)</option>
                  ))}
                </select>
                <button
                  className="btn btn-primary"
                  disabled={!trainFolderTarget || globalProgress?.status === "running" || globalProgress?.status === "starting"}
                  onClick={handleTrainFromFolder}
                >
                  🚀 Entraîner ce dossier
                </button>
              </div>
              {trainFolderTarget && folders[trainFolderTarget] && (
                <div style={{ marginTop: 10, fontSize: 12, color: C.muted }}>
                  Sessions : {folders[trainFolderTarget].map(s => s.name).join(", ")}
                </div>
              )}
            </div>
          )}

          {/* ─ Entraîner sessions manuelles ─ */}
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 14, padding: 24, marginBottom: 24 }}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>🎯 Sélection manuelle de sessions</div>
            <div style={{ fontSize: 12, color: C.muted, marginBottom: 16 }}>Cochez les sessions à inclure dans l'entraînement.</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 8, marginBottom: 16 }}>
              {sessions.map(s => {
                const hasSpeeds = s.images.some(i => i.status === "annotated");
                const isSel = trainSessions.has(s.name);
                return (
                  <label key={s.name} style={{
                    display: "flex", alignItems: "center", gap: 8,
                    background: isSel ? `${C.blue}20` : C.surface2,
                    border: `1px solid ${isSel ? C.blue : C.border}`,
                    borderRadius: 8, padding: "8px 12px", cursor: "pointer",
                    transition: "all 0.15s",
                  }}>
                    <input
                      type="checkbox"
                      checked={isSel}
                      onChange={e => {
                        const s2 = new Set(trainSessions);
                        if (e.target.checked) s2.add(s.name); else s2.delete(s.name);
                        setTrainSessions(s2);
                      }}
                    />
                    <span style={{ fontSize: 13, fontWeight: 600 }}>{s.name}</span>
                    {!hasSpeeds && <span style={{ fontSize: 10, color: C.orange, marginLeft: "auto" }}>!</span>}
                  </label>
                );
              })}
            </div>
            {/* Options */}
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16 }}>
              <div>
                <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 4 }}>Nom du modèle</label>
                <input
                  type="text" value={globalModelName}
                  onChange={e => setGlobalModelName(e.target.value)}
                  style={{ width: 200, padding: "6px 10px", background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, fontSize: 13 }}
                />
              </div>
              <div>
                <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 4 }}>Époques</label>
                <input
                  type="number" value={globalEpochs} min={5} max={500}
                  onChange={e => setGlobalEpochs(Number(e.target.value))}
                  style={{ width: 90, padding: "6px 10px", background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, fontSize: 13 }}
                />
              </div>
            </div>
            <button
              className="btn btn-primary"
              disabled={trainSessions.size === 0 || globalProgress?.status === "running" || globalProgress?.status === "starting"}
              onClick={() => handleLaunchGlobalTrain()}
            >
              🧠 Lancer l'entraînement ({trainSessions.size} session{trainSessions.size > 1 ? "s" : ""})
            </button>
          </div>

          {/* ─ Progression ─ */}
          {globalProgress && globalProgress.status !== "idle" && (
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 14, padding: 24, marginBottom: 24 }}>
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 12 }}>Progression entraînement</div>
              {(globalProgress.status === "running" || globalProgress.status === "starting") && (
                <>
                  <div style={{ height: 8, background: C.surface2, borderRadius: 4, overflow: "hidden", marginBottom: 8 }}>
                    <div style={{
                      height: "100%", borderRadius: 4,
                      width: `${globalProgress.total_epochs > 0 ? Math.round((globalProgress.epoch / globalProgress.total_epochs) * 100) : 0}%`,
                      background: `linear-gradient(90deg,${C.blue},${C.accent})`,
                      transition: "width 0.4s",
                    }} />
                  </div>
                  <div style={{ fontSize: 12, color: C.muted }}>
                    Époque {globalProgress.epoch} / {globalProgress.total_epochs}
                  </div>
                </>
              )}
              {globalProgress.status === "error" && (
                <div style={{ color: C.red, fontSize: 13 }}>❌ {(globalProgress as any).error}</div>
              )}
              {globalProgress.status === "done" && globalTrainResult && (
                <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
                  <div style={{ background: C.surface2, borderRadius: 10, padding: "12px 20px", textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 800, color: C.green }}>{globalTrainResult.mae?.toFixed(2)}</div>
                    <div style={{ fontSize: 12, color: C.muted }}>MAE (cm/s)</div>
                  </div>
                  <div style={{ background: C.surface2, borderRadius: 10, padding: "12px 20px", textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 800, color: C.blue }}>{globalTrainResult.rmse?.toFixed(2)}</div>
                    <div style={{ fontSize: 12, color: C.muted }}>RMSE (cm/s)</div>
                  </div>
                  <div style={{ background: C.surface2, borderRadius: 10, padding: "12px 20px", textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 800, color: C.accent }}>{globalTrainResult.n_train}</div>
                    <div style={{ fontSize: 12, color: C.muted }}>images train</div>
                  </div>
                  <div style={{ background: C.surface2, borderRadius: 10, padding: "12px 20px", textAlign: "center" }}>
                    <div style={{ fontSize: 22, fontWeight: 800, color: C.orange }}>{globalTrainResult.n_val}</div>
                    <div style={{ fontSize: 12, color: C.muted }}>images val</div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ─ Modèles disponibles + téléchargement ─ */}
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 14, padding: 24, marginBottom: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div style={{ fontWeight: 700, fontSize: 15 }}>💾 Modèles entraînés</div>
              <button className="btn btn-secondary" style={{ fontSize: 12 }} onClick={loadModels}>↻ Actualiser</button>
            </div>
            {models.length === 0 ? (
              <div style={{ textAlign: "center", padding: "24px 0", color: C.muted, fontSize: 13 }}>Aucun modèle disponible.</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {models.map(m => (
                  <div key={m.name} style={{
                    display: "flex", alignItems: "center", gap: 12,
                    background: C.surface2, borderRadius: 10, padding: "12px 16px",
                    border: `1px solid ${m.is_global ? C.accent + "40" : C.border}`,
                  }}>
                    <span style={{ fontSize: 20 }}>{m.is_global ? "🌍" : "📌"}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 700, fontSize: 13 }}>{m.name}</div>
                      <div style={{ fontSize: 11, color: C.muted }}>
                        {m.size_mb} Mo · {new Date(m.modified_at).toLocaleString("fr-FR")}
                      </div>
                    </div>
                    {/* Télécharger */}
                    <a
                      href={`http://localhost:8000/api/models/${m.filename}/download`}
                      download={m.filename}
                      className="btn btn-secondary"
                      style={{ fontSize: 12, padding: "5px 12px", textDecoration: "none", flexShrink: 0 }}
                    >
                      ⬇ Télécharger
                    </a>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* ─ Prédiction rapide ─ */}
          <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 14, padding: 24 }}>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 16 }}>🔍 Prédire Vc sur une image</div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
              <div style={{ flex: 1, minWidth: 200 }}>
                <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 6 }}>Modèle</label>
                <select
                  value={predictModel}
                  onChange={e => setPredictModel(e.target.value)}
                  style={{ width: "100%", padding: "8px 12px", background: C.surface2, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, fontSize: 13 }}
                >
                  <option value="">— Choisir —</option>
                  {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
                </select>
              </div>
              <div style={{ flex: 1, minWidth: 200 }}>
                <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 6 }}>Image</label>
                <input type="file" accept="image/*" onChange={e => setPredictFile(e.target.files?.[0] || null)}
                  style={{ fontSize: 12, color: C.muted }} />
              </div>
              <button className="btn btn-primary" onClick={handlePredict} disabled={!predictFile || !predictModel || predictLoading}>
                {predictLoading ? "Analyse..." : "Analyser"}
              </button>
            </div>
            {predictResult !== null && (
              <div style={{
                marginTop: 20, background: `linear-gradient(135deg, #0D2137, #0D1117)`,
                border: `2px solid ${C.accent}`, borderRadius: 14, padding: "24px 28px", textAlign: "center",
              }}>
                <div style={{ fontSize: 13, color: C.muted, marginBottom: 6 }}>Vitesse estimée</div>
                <div style={{ fontSize: 42, fontWeight: 900, color: C.accent, letterSpacing: "-1px" }}>
                  {predictResult} <span style={{ fontSize: 20, color: C.muted }}>cm/s</span>
                </div>
                <div style={{ fontSize: 12, color: C.muted, marginTop: 6 }}>Modèle : {predictModel}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── MODAL : Créer session ── */}
      {showCreate && (
        <div className="modal-backdrop" onClick={() => setShowCreate(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 420 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Nouvelle session</h2>
            <p style={{ fontSize: 13, color: C.muted, marginBottom: 20 }}>Nom sanitisé automatiquement (ASCII, underscores).</p>
            <div style={{ marginBottom: 14 }}>
              <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 6, fontWeight: 500 }}>Nom *</label>
              <input type="text" value={newName} onChange={e => handleNameChange(e.target.value)}
                placeholder="ex: canal_test_juin_2025" autoFocus style={{ width: "100%" }}
                onKeyDown={e => e.key === "Enter" && handleCreate()} />
            </div>
            <div style={{ marginBottom: 24 }}>
              <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 6, fontWeight: 500 }}>Description (optionnel)</label>
              <textarea value={newDesc} onChange={e => setNewDesc(e.target.value)}
                placeholder="Canal COSMER, courant 5–20 cm/s..." rows={3} style={{ width: "100%" }} />
            </div>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setShowCreate(false)}>Annuler</button>
              <button className="btn btn-primary" disabled={!newName.trim() || creating} onClick={handleCreate}>
                {creating ? "Création..." : "Créer la session"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── MODAL : Déplacer vers dossier ── */}
      {showMoveModal && (
        <div className="modal-backdrop" onClick={() => setShowMoveModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 400 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Déplacer vers un dossier</h2>
            <p style={{ fontSize: 13, color: C.muted, marginBottom: 20 }}>
              {selected.size} session{selected.size > 1 ? "s" : ""} sélectionnée{selected.size > 1 ? "s" : ""}
            </p>
            <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 6, fontWeight: 500 }}>Nom du dossier</label>
            <input
              type="text" value={moveFolderName} autoFocus
              onChange={e => setMoveFolderName(e.target.value)}
              placeholder="ex: Campagne_juillet_2025"
              list="existing-folders"
              style={{ width: "100%", marginBottom: 6 }}
            />
            <datalist id="existing-folders">
              {folderNames.map(f => <option key={f} value={f} />)}
            </datalist>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 20 }}>Laissez vide pour retirer du dossier.</div>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setShowMoveModal(false)}>Annuler</button>
              <button className="btn btn-primary" onClick={handleMoveToFolder}>Déplacer</button>
            </div>
          </div>
        </div>
      )}

      {/* ── MODAL : Créer dossier ── */}
      {showCreateFolder && (
        <div className="modal-backdrop" onClick={() => setShowCreateFolder(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 360 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Nouveau dossier</h2>
            <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 6, fontWeight: 500 }}>Nom du dossier</label>
            <input type="text" value={newFolderName} autoFocus onChange={e => setNewFolderName(e.target.value)}
              placeholder="ex: Campagne_juillet_2025" style={{ width: "100%", marginBottom: 20 }}
              onKeyDown={async e => {
                if (e.key === "Enter" && newFolderName.trim()) {
                  if (selected.size > 0) {
                    setMoveFolderName(newFolderName);
                    await batchMoveSessions(Array.from(selected), newFolderName);
                    setSelected(new Set());
                    await load();
                  }
                  setShowCreateFolder(false); setNewFolderName("");
                }
              }}
            />
            <p style={{ fontSize: 12, color: C.muted, marginBottom: 20 }}>Les sessions sélectionnées ({selected.size}) seront automatiquement déplacées dans ce dossier.</p>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setShowCreateFolder(false)}>Annuler</button>
              <button className="btn btn-primary" disabled={!newFolderName.trim()} onClick={async () => {
                if (selected.size > 0) {
                  await batchMoveSessions(Array.from(selected), newFolderName);
                  setSelected(new Set());
                  await load();
                } else {
                  // Dossier vide — juste noter pour usage futur
                  setFolders(p => ({ ...p, [newFolderName]: [] }));
                  setOpenFolders(p => ({ ...p, [newFolderName]: true }));
                }
                setShowCreateFolder(false); setNewFolderName("");
              }}>Créer</button>
            </div>
          </div>
        </div>
      )}

      {/* ── MODAL : Supprimer sélection ── */}
      {deleteSelected && (
        <div className="modal-backdrop" onClick={() => setDeleteSelected(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 380 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>Supprimer la sélection</h2>
            <p style={{ color: C.muted, fontSize: 14, marginBottom: 20 }}>
              Supprimer <strong style={{ color: C.red }}>{selected.size} session{selected.size > 1 ? "s" : ""}</strong> et toutes leurs données ? Irréversible.
            </p>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setDeleteSelected(false)}>Annuler</button>
              <button className="btn btn-red" onClick={handleDeleteSelected}>Supprimer tout</button>
            </div>
          </div>
        </div>
      )}

      {/* ── MODAL : Supprimer une session ── */}
      {deleteTarget && (
        <div className="modal-backdrop" onClick={() => setDeleteTarget(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 380 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>Supprimer la session</h2>
            <p style={{ color: C.muted, fontSize: 14, marginBottom: 20 }}>
              Supprimer <strong style={{ color: C.text }}>{deleteTarget}</strong> et toutes ses données ? Irréversible.
            </p>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setDeleteTarget(null)}>Annuler</button>
              <button className="btn btn-red" onClick={() => handleDelete(deleteTarget!)}>Supprimer</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
