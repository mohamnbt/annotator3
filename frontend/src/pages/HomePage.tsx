import { useEffect, useState, useCallback } from "react";
import { useLocation } from "wouter";
import {
  listSessions, createSession, deleteSession, sanitizeName,
  exportDownloadUrl,
  type SessionMeta,
} from "../lib/api";

const C = {
  bg: "#0D1117", surface: "#161B22", surface2: "#21262D",
  accent: "#00FFFF", green: "#00FF88", orange: "#FFA500",
  red: "#FF4444", blue: "#3B82F6", border: "#30363D",
  text: "#E6EDF3", muted: "#8B949E",
};

export default function HomePage() {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [, navigate] = useLocation();

  const load = useCallback(async () => {
    try {
      const data = await listSessions();
      setSessions(data);
    } catch {
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleNameChange = async (val: string) => {
    setNewName(val);
    if (val.length > 2) {
      const sanitized = await sanitizeName(val).catch(() => val);
      if (sanitized !== val) setNewName(sanitized);
    }
  };

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      await createSession(newName, newDesc);
      setShowCreate(false);
      setNewName(""); setNewDesc("");
      await load();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (name: string) => {
    try {
      await deleteSession(name);
      await load();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setDeleteTarget(null);
    }
  };

  const totalAnnotated = sessions.reduce((acc, s) => {
    return acc + s.images.filter((i) => i.status === "annotated").length;
  }, 0);

  return (
    <div className="animate-fade-in" style={{ maxWidth: 1100, margin: "0 auto", padding: "0 24px 60px" }}>
      {/* ── HERO HEADER ── */}
      <div style={{
        paddingTop: 40, paddingBottom: 32, borderBottom: `1px solid ${C.border}`, marginBottom: 32,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 10 }}>
              <span style={{ fontSize: 40 }}>🤿</span>
              <div>
                <h1 style={{
                  fontSize: 30, fontWeight: 900, letterSpacing: "-1px",
                  background: `linear-gradient(90deg, ${C.accent}, ${C.blue})`,
                  WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
                }}>COSMER Annotator</h1>
                <div style={{ fontSize: 13, color: C.muted, marginTop: 2 }}>
                  Laboratoire COSMER — Université de Toulon
                </div>
              </div>
            </div>
            <p style={{ color: C.muted, fontSize: 14, maxWidth: 520, lineHeight: 1.6 }}>
              Annotez des images de câbles de mouillage immergés pour estimer la
              <span style={{ color: C.accent, fontWeight: 600 }}> vitesse du courant marin Vc</span>{" "}
              grâce au deep learning.
            </p>
          </div>
          <div style={{ display: "flex", gap: 10, flexShrink: 0 }}>
            {totalAnnotated > 0 && (
              <a
                href="http://localhost:8000/api/export/global/download"
                className="btn btn-secondary"
                style={{ textDecoration: "none" }}
                download
              >
                🌍 Export global
              </a>
            )}
            <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
              + Nouvelle session
            </button>
          </div>
        </div>

        {/* Mini stats globales */}
        {sessions.length > 0 && (
          <div style={{ display: "flex", gap: 20, marginTop: 24 }}>
            {[
              { v: sessions.length, label: "Sessions", color: C.accent },
              { v: sessions.reduce((a, s) => a + s.images.length, 0), label: "Images totales", color: C.text },
              { v: totalAnnotated, label: "Annotées", color: C.green },
            ].map((s) => (
              <div key={s.label} style={{ fontSize: 13, color: C.muted }}>
                <span style={{ fontSize: 20, fontWeight: 800, color: s.color, marginRight: 6 }}>{s.v}</span>
                {s.label}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── SESSIONS ── */}
      {loading ? (
        <div style={{ textAlign: "center", padding: 60, color: C.muted }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>⚓</div>
          Chargement...
        </div>
      ) : sessions.length === 0 ? (
        <div style={{ textAlign: "center", padding: "60px 0" }}>
          <div style={{ fontSize: 52, marginBottom: 16, opacity: 0.3 }}>📂</div>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 8 }}>Aucune session</h2>
          <p style={{ color: C.muted, marginBottom: 24 }}>Créez votre première session pour commencer à annoter.</p>
          <button className="btn btn-primary" onClick={() => setShowCreate(true)}>+ Créer une session</button>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: 16 }}>
          {sessions.map((s) => {
            const ann = s.images.filter((i) => i.status === "annotated").length;
            const ign = s.images.filter((i) => i.status === "ignored").length;
            const total = s.images.length;
            const pct = total > 0 ? Math.round((ann / total) * 100) : 0;

            return (
              <div
                key={s.name}
                style={{
                  background: C.surface, border: `1px solid ${C.border}`,
                  borderRadius: 14, padding: 20, cursor: "pointer",
                  transition: "border-color 0.2s, transform 0.15s",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor = C.accent + "60";
                  (e.currentTarget as HTMLElement).style.transform = "translateY(-1px)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.borderColor = C.border;
                  (e.currentTarget as HTMLElement).style.transform = "";
                }}
                onClick={() => navigate(`/session/${s.name}`)}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 4 }}>{s.name}</div>
                    {s.description && (
                      <div style={{ fontSize: 12, color: C.muted, maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {s.description}
                      </div>
                    )}
                  </div>
                  <button
                    className="btn btn-red"
                    style={{ padding: "3px 8px", fontSize: 12, flexShrink: 0 }}
                    onClick={(e) => { e.stopPropagation(); setDeleteTarget(s.name); }}
                  >
                    🗑
                  </button>
                </div>

                {/* Progress */}
                <div style={{ marginBottom: 12 }}>
                  <div style={{ height: 6, background: C.surface2, borderRadius: 3, overflow: "hidden", marginBottom: 6 }}>
                    <div style={{
                      height: "100%", borderRadius: 3,
                      width: `${pct}%`,
                      background: pct === 100
                        ? `linear-gradient(90deg, ${C.green}, ${C.accent})`
                        : `linear-gradient(90deg, ${C.accent}, ${C.blue})`,
                      transition: "width 0.4s",
                    }} />
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: C.muted }}>
                    <span>{ann} annotées / {total}</span>
                    <span style={{ color: pct === 100 ? C.green : C.accent, fontWeight: 700 }}>{pct}%</span>
                  </div>
                </div>

                {/* Badges */}
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {ann > 0 && <span className="badge badge-green">{ann} annotées</span>}
                  {ign > 0 && <span className="badge badge-orange">{ign} ignorées</span>}
                  {total - ann - ign > 0 && <span className="badge badge-gray">{total - ann - ign} à annoter</span>}
                  {total === 0 && <span className="badge badge-gray">Vide</span>}
                </div>

                <div style={{ fontSize: 11, color: C.muted, marginTop: 12 }}>
                  Créé {new Date(s.created_at).toLocaleDateString("fr-FR")}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── MODAL CRÉER SESSION ── */}
      {showCreate && (
        <div className="modal-backdrop" onClick={() => setShowCreate(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 420 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>Nouvelle session</h2>
            <p style={{ fontSize: 13, color: C.muted, marginBottom: 20 }}>
              Le nom est automatiquement sanitisé (ASCII, underscores).
            </p>
            <div style={{ marginBottom: 14 }}>
              <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 6, fontWeight: 500 }}>Nom *</label>
              <input
                type="text"
                value={newName}
                onChange={(e) => handleNameChange(e.target.value)}
                placeholder="ex: canal_test_juin_2025"
                autoFocus
                style={{ width: "100%" }}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              />
            </div>
            <div style={{ marginBottom: 24 }}>
              <label style={{ fontSize: 12, color: C.muted, display: "block", marginBottom: 6, fontWeight: 500 }}>Description (optionnel)</label>
              <textarea
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="Canal COSMER, courant 5-20 cm/s, DJI Mini 3..."
                rows={3}
                style={{ width: "100%" }}
              />
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

      {/* ── MODAL SUPPRIMER ── */}
      {deleteTarget && (
        <div className="modal-backdrop" onClick={() => setDeleteTarget(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 380 }}>
            <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>Supprimer la session</h2>
            <p style={{ color: C.muted, fontSize: 14, marginBottom: 20 }}>
              Supprimer <strong style={{ color: C.text }}>{deleteTarget}</strong> et toutes ses images/annotations ? Cette action est irréversible.
            </p>
            <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setDeleteTarget(null)}>Annuler</button>
              <button className="btn btn-red" onClick={() => handleDelete(deleteTarget)}>Supprimer</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
