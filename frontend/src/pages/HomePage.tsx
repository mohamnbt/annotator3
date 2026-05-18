import { useEffect, useState, useCallback, useRef } from "react";
import { useLocation } from "wouter";
import {
  listSessions, createSession, deleteSession, sanitizeName,
  batchMoveSessions, listModels, bulkImportVideos, visualizeModel,
  exportBatchUrl,
  type SessionMeta, type ModelMeta,
} from "../lib/api";
import AIPanel from "../components/AIPanel";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

const C = {
  bg: "#0D1117", surface: "#161B22", surface2: "#21262D",
  accent: "#00FFFF", green: "#00FF88", orange: "#FFA500",
  red: "#FF4444", blue: "#3B82F6", border: "#30363D",
  text: "#E6EDF3", muted: "#8B949E",
};

type HomeTab = "sessions" | "ia";

interface BulkImportResult {
  name: string;
  ok: boolean;
  error?: string;
}

// ────────────────────────────────────────────────────────────────────────────
export default function HomePage() {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<HomeTab>("sessions");
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [newFolder, setNewFolder] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleteTargets, setDeleteTargets] = useState<string[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [openFolder, setOpenFolder] = useState<string | null>(null);
  const [showMoveModal, setShowMoveModal] = useState(false);
  const [moveTarget, setMoveTarget] = useState("");
  const [showFolderRename, setShowFolderRename] = useState<string | null>(null);
  const [folderRenameVal, setFolderRenameVal] = useState("");
  const [lasso, setLasso] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [lassoStart, setLassoStart] = useState<{ x: number; y: number } | null>(null);
  // ── Import multi-vidéos
  const [showBulkImport, setShowBulkImport] = useState(false);
  const [bulkFiles, setBulkFiles] = useState<File[]>([]);
  const [bulkFrameInterval, setBulkFrameInterval] = useState(120);
  const [bulkFolder, setBulkFolder] = useState("");
  const [bulkImporting, setBulkImporting] = useState(false);
  const [bulkProgress, setBulkProgress] = useState<{ done: number; total: number; current: string } | null>(null);
  const [bulkResults, setBulkResults] = useState<BulkImportResult[] | null>(null);
  const bulkInputRef = useRef<HTMLInputElement>(null);
  const [bulkDragOver, setBulkDragOver] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const [, navigate] = useLocation();

  const load = useCallback(async () => {
    try { setSessions(await listSessions()); }
    catch { setSessions([]); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  // ── Groupement par dossier
  const allFolders = Array.from(new Set(sessions.map(s => s.folder || ""))).filter(Boolean);
  const noFolder = sessions.filter(s => !s.folder);

  // ── Lasso
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest(".session-card, button, a, input, .modal-backdrop")) return;
    if (e.button !== 0) return;
    const rect = containerRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left + containerRef.current!.scrollLeft;
    const y = e.clientY - rect.top + containerRef.current!.scrollTop;
    setLassoStart({ x, y });
    setLasso({ x1: x, y1: y, x2: x, y2: y });
    setSelected(new Set());
  }, []);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!lassoStart || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left + containerRef.current.scrollLeft;
    const y = e.clientY - rect.top + containerRef.current.scrollTop;
    const newLasso = { x1: lassoStart.x, y1: lassoStart.y, x2: x, y2: y };
    setLasso(newLasso);
    const lx1 = Math.min(newLasso.x1, newLasso.x2);
    const ly1 = Math.min(newLasso.y1, newLasso.y2);
    const lx2 = Math.max(newLasso.x1, newLasso.x2);
    const ly2 = Math.max(newLasso.y1, newLasso.y2);
    const hit = new Set<string>();
    cardRefs.current.forEach((el, name) => {
      const r = el.getBoundingClientRect();
      const cr = containerRef.current!.getBoundingClientRect();
      const ex = r.left - cr.left + containerRef.current!.scrollLeft;
      const ey = r.top - cr.top + containerRef.current!.scrollTop;
      if (ex < lx2 && ex + r.width > lx1 && ey < ly2 && ey + r.height > ly1) hit.add(name);
    });
    setSelected(hit);
  }, [lassoStart]);

  const handleMouseUp = useCallback(() => {
    setLassoStart(null);
    setLasso(null);
  }, []);

  const toggleSelect = (name: string, e: React.MouseEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.stopPropagation();
      setSelected(prev => {
        const next = new Set(prev);
        if (next.has(name)) next.delete(name); else next.add(name);
        return next;
      });
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
      await createSession(newName, newDesc, newFolder);
      setShowCreate(false); setNewName(""); setNewDesc(""); setNewFolder("");
      await load();
    } catch (e: any) { alert(e.message); }
    finally { setCreating(false); }
  };

  const handleDeleteConfirm = async () => {
    for (const name of deleteTargets) await deleteSession(name).catch(() => {});
    setDeleteTargets([]);
    setSelected(new Set());
    await load();
  };

  const handleMove = async () => {
    if (selected.size === 0) return;
    await batchMoveSessions(Array.from(selected), moveTarget);
    setShowMoveModal(false);
    setSelected(new Set());
    setMoveTarget("");
    await load();
  };

  const handleFolderRename = async () => {
    if (!showFolderRename || !folderRenameVal.trim()) return;
    const affected = sessions.filter(s => s.folder === showFolderRename).map(s => s.name);
    await batchMoveSessions(affected, folderRenameVal.trim());
    setShowFolderRename(null);
    setFolderRenameVal("");
    await load();
  };

  // ── Export sélection
  const handleExportSelection = () => {
    const params = Array.from(selected).map(n => `sessions=${encodeURIComponent(n)}`).join("&");
    window.location.href = `http://localhost:8000/api/export/global/download?${params}`;
  };

  // ── Import multi-vidéos
  const handleBulkDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setBulkDragOver(false);
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("video/"));
    if (files.length) setBulkFiles(prev => [...prev, ...files]);
  };

  const handleBulkImport = async () => {
    if (!bulkFiles.length) return;
    setBulkImporting(true);
    setBulkResults(null);
    setBulkProgress({ done: 0, total: bulkFiles.length, current: "" });
    const results = await bulkImportVideos(
      bulkFiles,
      bulkFrameInterval,
      bulkFolder,
      (done, total, current) => setBulkProgress({ done, total, current }),
    );
    setBulkResults(results);
    setBulkImporting(false);
    await load();
  };

  const closeBulkImport = () => {
    if (bulkImporting) return;
    setShowBulkImport(false);
    setBulkFiles([]);
    setBulkResults(null);
    setBulkProgress(null);
    setBulkFolder("");
  };

  const totalAnnotated = sessions.reduce((a, s) =>
    a + s.images.filter(i => i.status === "annotated").length, 0);
  const selArray = Array.from(selected);

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px 60px", position: "relative" }}>

      {/* HEADER */}
      <div style={{ paddingTop: 40, paddingBottom: 24, borderBottom: `1px solid ${C.border}`, marginBottom: 28 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 8 }}>
              <span style={{ fontSize: 38 }}>🤿</span>
              <div>
                <h1 style={{
                  fontSize: 28, fontWeight: 900, letterSpacing: "-1px",
                  background: `linear-gradient(90deg, ${C.accent}, ${C.blue})`,
                  WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                }}>COSMER Annotator</h1>
                <div style={{ fontSize: 12, color: C.muted }}>Laboratoire COSMER — Université de Toulon</div>
              </div>
            </div>
            {sessions.length > 0 && (
              <div style={{ display: "flex", gap: 20, marginTop: 10 }}>
                {[
                  { v: sessions.length, l: "Sessions", c: C.accent },
                  { v: sessions.reduce((a, s) => a + s.images.length, 0), l: "Images", c: C.text },
                  { v: totalAnnotated, l: "Annotées", c: C.green },
                  { v: allFolders.length, l: "Dossiers", c: C.blue },
                ].map(s => (
                  <div key={s.l} style={{ fontSize: 13, color: C.muted }}>
                    <span style={{ fontSize: 18, fontWeight: 800, color: s.c, marginRight: 5 }}>{s.v}</span>{s.l}
                  </div>
                ))}
              </div>
            )}
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
            {totalAnnotated > 0 && (
              <a href="http://localhost:8000/api/export/global/download" className="btn btn-secondary" download style={{ textDecoration: "none" }}>
                🌍 Export global
              </a>
            )}
            {/* ── BOUTON IMPORT MULTI-VIDÉOS ── */}
            <button className="btn btn-secondary" onClick={() => setShowBulkImport(true)}
              style={{ display: "flex", alignItems: "center", gap: 6 }}>
              🎥 Importer des vidéos
            </button>
            <button className="btn btn-primary" onClick={() => setShowCreate(true)}>+ Nouvelle session</button>
          </div>
        </div>
      </div>

      {/* TABS */}
      <div style={{ display: "flex", gap: 2, marginBottom: 24, borderBottom: `1px solid ${C.border}` }}>
        {(["sessions", "ia"] as HomeTab[]).map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: "none", border: "none", cursor: "pointer",
            padding: "9px 18px", fontSize: 13, fontWeight: 600,
            color: tab === t ? C.accent : C.muted,
            borderBottom: tab === t ? `2px solid ${C.accent}` : "2px solid transparent",
            transition: "all 0.15s", display: "flex", alignItems: "center", gap: 6,
          }}>
            {t === "sessions" ? <>📂 Sessions</> : <>🧠 IA / Modèles</>}
          </button>
        ))}
      </div>

      {/* BARRE ACTIONS SÉLECTION */}
      {selected.size > 0 && tab === "sessions" && (
        <div className="animate-fade-in" style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 16px", marginBottom: 16,
          background: "rgba(0,255,255,0.06)",
          border: `1px solid rgba(0,255,255,0.25)`,
          borderRadius: 10,
        }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: C.accent }}>
            {selected.size} session{selected.size > 1 ? "s" : ""} sélectionnée{selected.size > 1 ? "s" : ""}
          </span>
          <button className="btn btn-secondary" style={{ fontSize: 12 }}
            onClick={() => { setShowMoveModal(true); setMoveTarget(""); }}>
            📂 Déplacer vers dossier
          </button>
<<<<<<< HEAD
          <button
            className="btn btn-secondary"
            style={{ fontSize: 12, color: C.green, borderColor: "rgba(0,255,136,0.4)" }}
            onClick={handleExportSelection}
          >
            ⬇ Export sélection
          </button>
=======
          <a href={exportBatchUrl(selArray)} className="btn btn-secondary" style={{ fontSize: 12, textDecoration: "none" }} download="batch_dataset.zip">
            📦 Exporter la sélection
          </a>
>>>>>>> daa8eb8 (Mon code actuel)
          <button className="btn btn-red" style={{ fontSize: 12 }}
            onClick={() => setDeleteTargets(selArray)}>
            🗑 Supprimer la sélection
          </button>
          <button className="btn btn-secondary" style={{ fontSize: 12, marginLeft: "auto" }}
            onClick={() => setSelected(new Set())}>
            ✕ Désélectionner
          </button>
        </div>
      )}

      {/* ONGLET SESSIONS */}
      {tab === "sessions" && (
        loading ? (
          <div style={{ textAlign: "center", padding: 60, color: C.muted }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>⚓</div>Chargement...
          </div>
        ) : sessions.length === 0 ? (
          <div style={{ textAlign: "center", padding: "60px 0" }}>
            <div style={{ fontSize: 52, marginBottom: 16, opacity: 0.3 }}>📂</div>
            <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>Aucune session</h2>
            <p style={{ color: C.muted, marginBottom: 24 }}>Créez une session ou importez des vidéos pour commencer.</p>
            <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
              <button className="btn btn-secondary" onClick={() => setShowBulkImport(true)}>🎥 Importer des vidéos</button>
              <button className="btn btn-primary" onClick={() => setShowCreate(true)}>+ Créer une session</button>
            </div>
          </div>
        ) : (
          <div
            ref={containerRef}
            style={{ position: "relative", userSelect: "none", minHeight: 300 }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {lasso && (() => {
              const x = Math.min(lasso.x1, lasso.x2);
              const y = Math.min(lasso.y1, lasso.y2);
              const w = Math.abs(lasso.x2 - lasso.x1);
              const h = Math.abs(lasso.y2 - lasso.y1);
              return (
                <div style={{
                  position: "absolute", left: x, top: y, width: w, height: h,
                  border: `1.5px dashed ${C.accent}`,
                  background: "rgba(0,255,255,0.06)",
                  borderRadius: 4, pointerEvents: "none", zIndex: 50,
                }} />
              );
            })()}

            {allFolders.map(folder => {
              const folderSessions = sessions.filter(s => s.folder === folder);
              const isOpen = openFolder === folder;
              return (
                <div key={folder} style={{ marginBottom: 20 }}>
                  <div style={{
                    display: "flex", alignItems: "center", gap: 10,
                    padding: "10px 16px", cursor: "pointer",
                    background: C.surface, border: `1px solid ${C.border}`,
                    borderRadius: isOpen ? "10px 10px 0 0" : 10, transition: "all 0.15s",
                  }}
                    onClick={() => setOpenFolder(isOpen ? null : folder)}
                  >
                    <span style={{ fontSize: 18 }}>{isOpen ? "📂" : "📁"}</span>
                    <span style={{ fontWeight: 700, fontSize: 15, flex: 1 }}>{folder}</span>
                    <span style={{ fontSize: 12, color: C.muted }}>
                      {folderSessions.length} session{folderSessions.length > 1 ? "s" : ""}{" — "}
                      {folderSessions.reduce((a, s) => a + s.images.filter(i => i.status === "annotated").length, 0)} annotées
                    </span>
                    <button className="btn btn-secondary" style={{ fontSize: 11, padding: "2px 8px" }}
                      onClick={e => { e.stopPropagation(); setShowFolderRename(folder); setFolderRenameVal(folder); }}
                    >Renommer</button>
                    <button className="btn btn-red" style={{ fontSize: 11, padding: "2px 8px" }}
                      onClick={e => { e.stopPropagation(); setDeleteTargets(folderSessions.map(s => s.name)); }}
                    >🗑 Tout supprimer</button>
                    <button className="btn btn-secondary" style={{ fontSize: 11, padding: "2px 8px" }}
                      onClick={e => {
                        e.stopPropagation();
                        const names = new Set(folderSessions.map(s => s.name));
                        setSelected(prev => {
                          const allIn = folderSessions.every(s => prev.has(s.name));
                          if (allIn) { const n = new Set(prev); folderSessions.forEach(s => n.delete(s.name)); return n; }
                          return new Set([...prev, ...names]);
                        });
                      }}
                    >Sélectionner</button>
                    <span style={{ fontSize: 14, color: C.muted, marginLeft: 4 }}>{isOpen ? "▲" : "▼"}</span>
                  </div>
                  {isOpen && (
                    <div style={{
                      background: "rgba(22,27,34,0.6)",
                      border: `1px solid ${C.border}`, borderTop: "none",
                      borderRadius: "0 0 10px 10px", padding: 16,
                    }}>
                      <SessionGrid sessions={folderSessions} selected={selected} cardRefs={cardRefs}
                        onOpen={n => navigate(`/session/${n}`)}
                        onDelete={n => setDeleteTargets([n])}
                        onToggle={toggleSelect} />
                    </div>
                  )}
                </div>
              );
            })}

            {noFolder.length > 0 && (
              <div style={{ marginTop: allFolders.length > 0 ? 8 : 0 }}>
                {allFolders.length > 0 && (
                  <div style={{ fontSize: 12, color: C.muted, marginBottom: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>
                    Sans dossier
                  </div>
                )}
                <SessionGrid sessions={noFolder} selected={selected} cardRefs={cardRefs}
                  onOpen={n => navigate(`/session/${n}`)}
                  onDelete={n => setDeleteTargets([n])}
                  onToggle={toggleSelect} />
              </div>
            )}
          </div>
        )
      )}

      {/* ONGLET IA */}
      {tab === "ia" && (
        <div className="animate-fade-in">
          <div style={{
            background: "linear-gradient(135deg, rgba(0,255,255,0.06), rgba(59,130,246,0.06))",
            border: `1px solid rgba(0,255,255,0.2)`, borderRadius: 14, padding: "18px 24px",
            marginBottom: 24, display: "flex", alignItems: "center", gap: 16,
          }}>
            <span style={{ fontSize: 36 }}>🌍</span>
            <div>
              <div style={{ fontSize: 16, fontWeight: 700, color: C.accent }}>Entraînement global Vc</div>
              <div style={{ fontSize: 13, color: C.muted, marginTop: 3 }}>
                Combinez plusieurs sessions (ou dossiers entiers) pour entraîner un modèle Vc robuste.
              </div>
            </div>
          </div>
          <GlobalAITrainer sessions={sessions} allFolders={allFolders} />
        </div>
      )}

      {/* ── MODAL IMPORT MULTI-VIDÉOS ── */}
      {showBulkImport && (
        <div className="modal-backdrop" onClick={closeBulkImport}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 560, maxHeight: "85vh", overflowY: "auto" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
              <span style={{ fontSize: 26 }}>🎥</span>
              <h2 style={{ fontSize: 18, fontWeight: 700 }}>Import de vidéos en lot</h2>
            </div>
            <p style={{ fontSize: 13, color: C.muted, marginBottom: 20, lineHeight: 1.6 }}>
              Chaque vidéo crée automatiquement une session portant le <strong style={{ color: C.text }}>nom du fichier</strong>.
              Les frames sont extraites en arrière-plan — vous pouvez annuler la fenêtre sans interrompre l'extraction.
            </p>

            {/* Zone de dépôt */}
            {!bulkResults && (
              <div
                onDragOver={e => { e.preventDefault(); setBulkDragOver(true); }}
                onDragLeave={() => setBulkDragOver(false)}
                onDrop={handleBulkDrop}
                onClick={() => bulkInputRef.current?.click()}
                style={{
                  border: `2px dashed ${bulkDragOver ? C.accent : C.border}`,
                  borderRadius: 12, padding: "24px 20px",
                  textAlign: "center", cursor: "pointer",
                  background: bulkDragOver ? "rgba(0,255,255,0.04)" : "transparent",
                  transition: "all 0.15s", marginBottom: 16,
                }}
              >
                <input
                  ref={bulkInputRef}
                  type="file"
                  accept="video/*"
                  multiple
                  hidden
                  onChange={e => {
                    const files = Array.from(e.target.files || []);
                    if (files.length) setBulkFiles(prev => [...prev, ...files]);
                    e.target.value = "";
                  }}
                />
                <div style={{ fontSize: 32, marginBottom: 8 }}>🎥</div>
                <div style={{ fontSize: 14, fontWeight: 600, color: C.text }}>Glissez vos vidéos ici</div>
                <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>ou cliquez pour parcourir — MP4, MOV, AVI, MKV...</div>
              </div>
            )}

            {/* Liste des vidéos sélectionnées */}
            {bulkFiles.length > 0 && !bulkResults && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 12, color: C.muted, fontWeight: 600, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
                  {bulkFiles.length} vidéo{bulkFiles.length > 1 ? "s" : ""} sélectionnée{bulkFiles.length > 1 ? "s" : ""}
                </div>
                <div style={{ maxHeight: 180, overflowY: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
                  {bulkFiles.map((f, i) => {
                    const sessionName = f.name.replace(/\.[^.]+$/, "");
                    return (
                      <div key={i} style={{
                        display: "flex", alignItems: "center", justifyContent: "space-between",
                        padding: "8px 12px", background: C.surface2,
                        borderRadius: 8, border: `1px solid ${C.border}`,
                      }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 0 }}>
                          <span style={{ fontSize: 16 }}>🎥</span>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: 13, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.name}</div>
                            <div style={{ fontSize: 11, color: C.muted }}>
                              Session : <span style={{ color: C.accent }}>{sessionName}</span>
                              {" — "}{(f.size / 1024 / 1024).toFixed(1)} MB
                            </div>
                          </div>
                        </div>
                        {!bulkImporting && (
                          <button className="btn btn-red" style={{ fontSize: 11, padding: "2px 8px", flexShrink: 0 }}
                            onClick={() => setBulkFiles(prev => prev.filter((_, j) => j !== i))}>
                            ✕
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Options */}
            {!bulkResults && (
              <div style={{ display: "flex", flexDirection: "column", gap: 12, marginBottom: 20 }}>
                <Field label="Intervalle d'extraction (1 frame toutes les N frames)">
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input type="number" value={bulkFrameInterval} min={10} max={3000}
                      onChange={e => setBulkFrameInterval(parseInt(e.target.value) || 120)}
                      style={{ width: 80 }} />
                    <span style={{ fontSize: 11, color: C.muted }}>
                      {bulkFrameInterval} frames ≈ {(bulkFrameInterval / 30).toFixed(1)}s à 30fps
                    </span>
                  </div>
                </Field>
                <Field label="Ranger dans un dossier (optionnel)">
                  <input type="text" value={bulkFolder}
                    onChange={e => setBulkFolder(e.target.value)}
                    placeholder="ex: Campagne_Mai_2025"
                    list="bulk-folder-suggestions" />
                  <datalist id="bulk-folder-suggestions">
                    {allFolders.map(f => <option key={f} value={f} />)}
                  </datalist>
                </Field>
              </div>
            )}

            {/* Progression */}
            {bulkProgress && !bulkResults && (
              <div style={{
                background: C.surface2, borderRadius: 10, padding: 16,
                border: `1px solid ${C.border}`, marginBottom: 16,
              }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: 13 }}>
                  <span style={{ color: C.text }}>
                    {bulkImporting
                      ? `Envoi en cours… ${bulkProgress.current ? `(${bulkProgress.current})` : ""}`
                      : "Import terminé"}
                  </span>
                  <span style={{ color: C.accent, fontWeight: 700 }}>{bulkProgress.done}/{bulkProgress.total}</span>
                </div>
                <div style={{ background: C.border, borderRadius: 4, height: 6, overflow: "hidden" }}>
                  <div style={{
                    height: "100%", borderRadius: 4,
                    width: `${Math.round((bulkProgress.done / bulkProgress.total) * 100)}%`,
                    background: `linear-gradient(90deg, ${C.accent}, ${C.blue})`,
                    transition: "width 0.3s",
                  }} />
                </div>
                <div style={{ fontSize: 11, color: C.muted, marginTop: 6 }}>
                  L'extraction des frames continue en arrière-plan même après fermeture.
                </div>
              </div>
            )}

            {/* Résultats */}
            {bulkResults && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12, color: C.text }}>
                  ✅ {bulkResults.filter(r => r.ok).length} session{bulkResults.filter(r => r.ok).length > 1 ? "s" : ""} créée{bulkResults.filter(r => r.ok).length > 1 ? "s" : ""}
                  {bulkResults.some(r => !r.ok) && (
                    <span style={{ color: C.red, marginLeft: 12 }}>
                      ❌ {bulkResults.filter(r => !r.ok).length} erreur{bulkResults.filter(r => !r.ok).length > 1 ? "s" : ""}
                    </span>
                  )}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {bulkResults.map((r, i) => (
                    <div key={i} style={{
                      display: "flex", alignItems: "center", gap: 10,
                      padding: "8px 12px", borderRadius: 8,
                      background: r.ok ? "rgba(0,255,136,0.06)" : "rgba(255,68,68,0.06)",
                      border: `1px solid ${r.ok ? "rgba(0,255,136,0.2)" : "rgba(255,68,68,0.2)"}`,
                    }}>
                      <span style={{ fontSize: 16 }}>{r.ok ? "✅" : "❌"}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: r.ok ? C.green : C.red }}>{r.name}</div>
                        {r.error && <div style={{ fontSize: 11, color: C.muted }}>{r.error}</div>}
                        {r.ok && <div style={{ fontSize: 11, color: C.muted }}>Extraction des frames en cours en arrière-plan…</div>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={closeBulkImport} disabled={bulkImporting}>
                {bulkResults ? "Fermer" : "Annuler"}
              </button>
              {!bulkResults && (
                <button className="btn btn-primary"
                  disabled={!bulkFiles.length || bulkImporting}
                  onClick={handleBulkImport}>
                  {bulkImporting
                    ? `⏳ Import... ${bulkProgress ? `${bulkProgress.done}/${bulkProgress.total}` : ""}`
                    : `🎥 Importer ${bulkFiles.length} vidéo${bulkFiles.length > 1 ? "s" : ""}`}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* MODAL CRÉER SESSION */}
      {showCreate && (
        <div className="modal-backdrop" onClick={() => setShowCreate(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 440 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Nouvelle session</h2>
            <p style={{ fontSize: 13, color: C.muted, marginBottom: 20 }}>Le nom est automatiquement sanitiisé (ASCII, underscores).</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <Field label="Nom *">
                <input type="text" value={newName} onChange={e => handleNameChange(e.target.value)}
                  placeholder="ex: canal_test_juin_2025" autoFocus
                  onKeyDown={e => e.key === "Enter" && handleCreate()} />
              </Field>
              <Field label="Description (optionnel)">
                <textarea value={newDesc} onChange={e => setNewDesc(e.target.value)}
                  placeholder="DJI Mini 3, canal COSMER..." rows={2} />
              </Field>
              <Field label="Dossier (optionnel)">
                <input type="text" value={newFolder} onChange={e => setNewFolder(e.target.value)}
                  placeholder="ex: Campagne_Mai_2025" list="folder-suggestions" />
                <datalist id="folder-suggestions">
                  {allFolders.map(f => <option key={f} value={f} />)}
                </datalist>
              </Field>
            </div>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end", marginTop: 20 }}>
              <button className="btn btn-secondary" onClick={() => setShowCreate(false)}>Annuler</button>
              <button className="btn btn-primary" disabled={!newName.trim() || creating} onClick={handleCreate}>
                {creating ? "Création..." : "Créer"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL SUPPRIMER */}
      {deleteTargets.length > 0 && (
        <div className="modal-backdrop" onClick={() => setDeleteTargets([])}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 400 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>Supprimer</h2>
            <p style={{ color: C.muted, fontSize: 14, marginBottom: 8 }}>
              Supprimer <strong style={{ color: C.red }}>{deleteTargets.length} session{deleteTargets.length > 1 ? "s" : ""}</strong> et toutes leurs données ? Irréversible.
            </p>
            {deleteTargets.length <= 5 && (
              <ul style={{ fontSize: 13, color: C.muted, marginBottom: 16, paddingLeft: 16 }}>
                {deleteTargets.map(n => <li key={n}>{n}</li>)}
              </ul>
            )}
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setDeleteTargets([])}>Annuler</button>
              <button className="btn btn-red" onClick={handleDeleteConfirm}>Supprimer</button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL DÉPLACER */}
      {showMoveModal && (
        <div className="modal-backdrop" onClick={() => setShowMoveModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 400 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Déplacer vers un dossier</h2>
            <p style={{ color: C.muted, fontSize: 13, marginBottom: 16 }}>
              {selected.size} session{selected.size > 1 ? "s" : ""} sélectionnée{selected.size > 1 ? "s" : ""}
            </p>
            <input type="text" value={moveTarget} onChange={e => setMoveTarget(e.target.value)}
              placeholder="Nom du dossier (vide = sans dossier)"
              list="folder-suggestions-move" autoFocus />
            <datalist id="folder-suggestions-move">
              {allFolders.map(f => <option key={f} value={f} />)}
              <option value="" />
            </datalist>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end", marginTop: 16 }}>
              <button className="btn btn-secondary" onClick={() => setShowMoveModal(false)}>Annuler</button>
              <button className="btn btn-primary" onClick={handleMove}>📂 Déplacer</button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL RENOMMER DOSSIER */}
      {showFolderRename && (
        <div className="modal-backdrop" onClick={() => setShowFolderRename(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: 380 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12 }}>Renommer le dossier</h2>
            <input type="text" value={folderRenameVal} onChange={e => setFolderRenameVal(e.target.value)}
              autoFocus onKeyDown={e => e.key === "Enter" && handleFolderRename()} />
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end", marginTop: 16 }}>
              <button className="btn btn-secondary" onClick={() => setShowFolderRename(null)}>Annuler</button>
              <button className="btn btn-primary" onClick={handleFolderRename}>Renommer</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
function SessionGrid({
  sessions, selected, cardRefs, onOpen, onDelete, onToggle,
}: {
  sessions: SessionMeta[];
  selected: Set<string>;
  cardRefs: React.MutableRefObject<Map<string, HTMLDivElement>>;
  onOpen: (name: string) => void;
  onDelete: (name: string) => void;
  onToggle: (name: string, e: React.MouseEvent) => void;
}) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 14 }}>
      {sessions.map(s => {
        const ann = s.images.filter(i => i.status === "annotated").length;
        const ign = s.images.filter(i => i.status === "ignored").length;
        const total = s.images.length;
        const pct = total > 0 ? Math.round((ann / total) * 100) : 0;
        const isSel = selected.has(s.name);
        return (
          <div
            key={s.name}
            className="session-card"
            ref={el => { if (el) cardRefs.current.set(s.name, el); else cardRefs.current.delete(s.name); }}
            style={{
              background: isSel ? "rgba(0,255,255,0.07)" : "#161B22",
              border: isSel ? `2px solid ${"#00FFFF"}` : "1px solid #30363D",
              borderRadius: 12, padding: 18, cursor: "pointer",
              transition: "all 0.15s",
              transform: isSel ? "scale(1.015)" : "",
            }}
            onClick={e => {
              if (e.ctrlKey || e.metaKey) { onToggle(s.name, e); return; }
              onOpen(s.name);
            }}
            onMouseEnter={e => { if (!isSel) (e.currentTarget as HTMLElement).style.borderColor = "rgba(0,255,255,0.4)"; }}
            onMouseLeave={e => { if (!isSel) (e.currentTarget as HTMLElement).style.borderColor = "#30363D"; }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 3 }}>
                  {isSel && <span style={{ color: "#00FFFF", fontSize: 14 }}>✓</span>}
                  <span style={{ fontWeight: 700, fontSize: 15, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {s.name}
                  </span>
                </div>
                {s.description && (
                  <div style={{ fontSize: 12, color: "#8B949E", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {s.description}
                  </div>
                )}
              </div>
              <button
                className="btn btn-red"
                style={{ padding: "3px 8px", fontSize: 11, flexShrink: 0, marginLeft: 8 }}
                onClick={e => { e.stopPropagation(); onDelete(s.name); }}
              >🗑</button>
            </div>
            <div style={{ marginBottom: 10 }}>
              <div style={{ height: 5, background: "#21262D", borderRadius: 3, overflow: "hidden", marginBottom: 5 }}>
                <div style={{
                  height: "100%", borderRadius: 3, width: `${pct}%`,
                  background: pct === 100
                    ? "linear-gradient(90deg, #00FF88, #00FFFF)"
                    : "linear-gradient(90deg, #00FFFF, #3B82F6)",
                  transition: "width 0.4s",
                }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#8B949E" }}>
                <span>{ann} / {total} annotées</span>
                <span style={{ color: pct === 100 ? "#00FF88" : "#00FFFF", fontWeight: 700 }}>{pct}%</span>
              </div>
            </div>
            <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
              {ann > 0 && <span className="badge badge-green">{ann} ✓</span>}
              {ign > 0 && <span className="badge badge-orange">{ign} ign.</span>}
              {total - ann - ign > 0 && <span className="badge badge-gray">{total - ann - ign} restantes</span>}
              {total === 0 && <span className="badge badge-gray">Vide</span>}
            </div>
            <div style={{ fontSize: 10, color: "#8B949E", marginTop: 10 }}>
              {new Date(s.created_at).toLocaleDateString("fr-FR")}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
function GlobalAITrainer({ sessions, allFolders }: { sessions: SessionMeta[]; allFolders: string[] }) {
  const [mode, setMode] = useState<"folder" | "manual">("folder");
  const [selFolder, setSelFolder] = useState("");
  const [manualSessions, setManualSessions] = useState<Set<string>>(new Set());
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [vizModel, setVizModel] = useState<string | null>(null);
  const [vizData, setVizData] = useState<{ theta: number; vc_real: number; vc_pred: number }[]>([]);
  const [vizLoading, setVizLoading] = useState(false);

  useEffect(() => { listModels().then(setModels).catch(() => {}); }, []);

  const toggleSession = (name: string) => {
    setManualSessions(prev => { const n = new Set(prev); if (n.has(name)) n.delete(name); else n.add(name); return n; });
  };

  const handleVisualize = async (modelName: string) => {
    if (vizModel === modelName) { setVizModel(null); setVizData([]); return; }
    setVizModel(modelName);
    setVizLoading(true);
    try {
      const d = await visualizeModel(modelName);
      if (Array.isArray(d)) setVizData(d);
    } catch {
      setVizData([]);
    } finally {
      setVizLoading(false);
    }
  };

  const getSessionsForTraining = () => {
    if (mode === "folder" && selFolder) return sessions.filter(s => s.folder === selFolder).map(s => s.name).join(",");
    return Array.from(manualSessions).join(",");
  };

  const trainSessions = getSessionsForTraining();

  // ── Logic de visualisation ────────────────────────────────────────────────
  const vizPtsReal = vizData.map(d => ({ x: d.theta, y: d.vc_real }));
  const vizPtsPred = vizData.map(d => ({ x: d.theta, y: d.vc_pred }));

  const allVizTh = vizData.map(d => d.theta);
  const vizThMin = allVizTh.length > 0 ? Math.min(...allVizTh) - 2 : 0;
  const vizThMax = allVizTh.length > 0 ? Math.max(...allVizTh) + 2 : 90;
  const allVizVc = vizData.flatMap(d => [d.vc_real, d.vc_pred]);
  const vizVcMin = allVizVc.length > 0 ? Math.min(...allVizVc) - 2 : 0;
  const vizVcMax = allVizVc.length > 0 ? Math.max(...allVizVc) + 2 : 40;

  const TS = { background: "#161B22", border: "1px solid #30363D", borderRadius: 8, fontSize: 11 };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ background: "#161B22", border: "1px solid #30363D", borderRadius: 14, overflow: "hidden" }}>
        <div style={{ padding: "14px 20px", borderBottom: "1px solid #30363D", display: "flex", alignItems: "center", gap: 10, background: "rgba(255,255,255,0.02)" }}>
          <span style={{ fontSize: 18 }}>🎯</span>
          <span style={{ fontWeight: 700, fontSize: 15, color: "#E6EDF3" }}>Choisir les données d'entraînement</span>
        </div>
        <div style={{ padding: 20 }}>
          <div style={{ display: "flex", gap: 10, marginBottom: 20 }}>
            {(["folder", "manual"] as const).map(m => (
              <button key={m} onClick={() => setMode(m)} className={mode === m ? "btn btn-primary" : "btn btn-secondary"} style={{ fontSize: 12 }}>
                {m === "folder" ? "📂 Par dossier" : "☰ Sélection manuelle"}
              </button>
            ))}
          </div>
          {mode === "folder" && (
            <div>
              {allFolders.length === 0 ? (
                <p style={{ color: "#8B949E", fontSize: 13 }}>Aucun dossier créé.</p>
              ) : (
                <>
                  <label style={{ fontSize: 12, color: "#8B949E", display: "block", marginBottom: 6, fontWeight: 500 }}>Dossier source</label>
                  <select value={selFolder} onChange={e => setSelFolder(e.target.value)} style={{ width: 300, marginBottom: 12 }}>
                    <option value="">-- Sélectionner un dossier --</option>
                    {allFolders.map(f => <option key={f} value={f}>{f} ({sessions.filter(s => s.folder === f).length} sessions)</option>)}
                  </select>
                  {selFolder && (
                    <div style={{ fontSize: 13, color: "#8B949E" }}>
                      {sessions.filter(s => s.folder === selFolder).map(s => (
                        <span key={s.name} style={{ display: "inline-block", margin: "2px 4px", padding: "2px 10px", background: "rgba(0,255,255,0.07)", border: "1px solid rgba(0,255,255,0.2)", borderRadius: 20, fontSize: 11, color: "#00FFFF" }}>{s.name}</span>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
          {mode === "manual" && (
            <div>
              <label style={{ fontSize: 12, color: "#8B949E", display: "block", marginBottom: 10, fontWeight: 500 }}>Cliquez pour sélectionner / désélectionner</label>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 8 }}>
                {sessions.map(s => {
                  const ann = s.images.filter(i => i.status === "annotated").length;
                  const isSel = manualSessions.has(s.name);
                  return (
                    <div key={s.name} onClick={() => toggleSession(s.name)} style={{ padding: "10px 14px", borderRadius: 8, cursor: "pointer", background: isSel ? "rgba(0,255,255,0.08)" : "#0D1117", border: isSel ? "1.5px solid #00FFFF" : "1px solid #30363D", transition: "all 0.15s" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        {isSel && <span style={{ color: "#00FFFF" }}>✓</span>}
                        <span style={{ fontSize: 13, fontWeight: 600, color: isSel ? "#00FFFF" : "#E6EDF3" }}>{s.name}</span>
                      </div>
                      <div style={{ fontSize: 11, color: "#8B949E", marginTop: 3 }}>{ann} ann.</div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      <AIPanel sessionName="_global" isGlobal preselectedSessions={trainSessions} />

      {models.length > 0 && (
        <div style={{ background: "#161B22", border: "1px solid #30363D", borderRadius: 14, overflow: "hidden" }}>
          <div style={{ padding: "14px 20px", borderBottom: "1px solid #30363D", background: "rgba(255,255,255,0.02)", display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 18 }}>📦</span>
            <span style={{ fontWeight: 700, fontSize: 15, color: "#E6EDF3" }}>Modèles disponibles</span>
          </div>
          <div style={{ padding: 20 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {models.map(m => (
                <div key={m.name}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 16px", background: "#0D1117", borderRadius: vizModel === m.name ? "10px 10px 0 0" : 10, border: vizModel === m.name ? "1px solid #00FFFF" : "1px solid #30363D" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ fontSize: 18 }}>{m.is_global ? "🌍" : "📂"}</span>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: 14 }}>{m.name}</div>
                        <div style={{ fontSize: 11, color: "#8B949E" }}>{m.size_mb} MB — {new Date(m.modified_at).toLocaleDateString("fr-FR")}</div>
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 8 }}>
                      <button className="btn btn-secondary" style={{ fontSize: 12 }} onClick={() => handleVisualize(m.name)}>
                        {vizLoading && vizModel === m.name ? "⏳..." : vizModel === m.name ? "✕ Fermer" : "📈 Visualiser"}
                      </button>
                      <a href={`http://localhost:8000/api/models/${encodeURIComponent(m.name)}/training-script`} className="btn btn-secondary" style={{ fontSize: 12, textDecoration: "none" }} download={`train_${m.name}.py`}>
                        📜 Script
                      </a>
                      <a href={`http://localhost:8000/api/models/${encodeURIComponent(m.filename)}/download`} className="btn btn-secondary" style={{ fontSize: 12, textDecoration: "none" }} download={m.filename}>
                        ⬇ Modèle
                      </a>
                    </div>
                  </div>

                  {vizModel === m.name && (
                    <div style={{ padding: 20, background: "rgba(0,255,255,0.02)", border: "1px solid #00FFFF", borderTop: "none", borderRadius: "0 0 10px 10px" }}>
                      {vizLoading ? (
                        <div style={{ textAlign: "center", color: "#8B949E", padding: 20 }}>⏳ Calcul des prédictions...</div>
                      ) : vizData.length === 0 ? (
                        <div style={{ textAlign: "center", color: "#8B949E", padding: 20 }}>Pas de données d'annotation pour ce modèle.</div>
                      ) : (
                        <>
                          <div style={{ display: "flex", gap: 20, marginBottom: 16, fontSize: 11 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                              <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#00FFFF" }} />
                              <span style={{ color: "#E6EDF3" }}>Réel (Annotations)</span>
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                              <div style={{ width: 10, height: 10, borderRadius: "50%", background: "#00FF88" }} />
                              <span style={{ color: "#E6EDF3" }}>Modèle (Prédictions)</span>
                            </div>
                          </div>
                          <ResponsiveContainer width="100%" height={260}>
                            <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#30363D" />
                              <XAxis type="number" dataKey="x" name="θ" unit="°" domain={[vizThMin, vizThMax]} tick={{ fontSize: 10, fill: "#8B949E" }} />
                              <YAxis type="number" dataKey="y" name="Vc" unit="cm/s" domain={[vizVcMin, vizVcMax]} tick={{ fontSize: 10, fill: "#8B949E" }} />
                              <Tooltip contentStyle={TS} />
                              <Scatter name="Réel" data={vizPtsReal} fill="#00FFFF" opacity={0.6} />
                              <Scatter name="Prédit" data={vizPtsPred} fill="#00FF88" opacity={0.8} />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label style={{ fontSize: 12, color: "#8B949E", display: "block", marginBottom: 5, fontWeight: 500 }}>{label}</label>
      {children}
    </div>
  );
}
