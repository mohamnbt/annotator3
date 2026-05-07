import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { listSessions, createSession, deleteSession, sanitizeName, type SessionMeta } from "../lib/api";

export default function HomePage() {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [sanitized, setSanitized] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [creating, setCreating] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [, navigate] = useLocation();

  const load = async () => {
    try {
      const data = await listSessions();
      setSessions(data);
    } catch {
      /* empty */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  useEffect(() => {
    if (!newName) { setSanitized(""); return; }
    const t = setTimeout(async () => {
      const s = await sanitizeName(newName);
      setSanitized(s);
    }, 100);
    return () => clearTimeout(t);
  }, [newName]);

  const handleCreate = async () => {
    if (!sanitized) return;
    setCreating(true);
    try {
      await createSession(newName, newDesc);
      setShowCreate(false);
      setNewName("");
      setNewDesc("");
      await load();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;
    try {
      await deleteSession(deleteTarget);
      setDeleteTarget(null);
      await load();
    } catch (e: any) {
      alert(e.message);
    }
  };

  const getStats = (session: SessionMeta) => {
    const total = session.images?.length || 0;
    const annotated = session.images?.filter((i) => i.status === "annotated").length || 0;
    const ignored = session.images?.filter((i) => i.status === "ignored").length || 0;
    const pct = total > 0 ? Math.round((annotated / total) * 100) : 0;
    return { total, annotated, ignored, pct };
  };

  const handleDownloadMerged = () => {
    window.location.href = "http://localhost:8000/api/export/merged/download";
  };

  return (
    <div className="animate-fade-in">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 32 }}>
        <div>
          <h1 style={{ fontSize: 28, fontWeight: 700, marginBottom: 4 }}>Sessions d'annotation</h1>
          <p style={{ color: "var(--color-text-dim)", fontSize: 14 }}>
            Gérez vos sessions de données pour l'entraînement YOLOv8 segmentation
          </p>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button
            className="btn btn-secondary"
            onClick={handleDownloadMerged}
            title="Télécharger toutes les sessions annotées en un seul dataset fusionné"
          >
            ⬇️ Tout fusionné
          </button>
          <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
            <span style={{ fontSize: 18 }}>+</span> Nouvelle session
          </button>
        </div>
      </div>

      {loading ? (
        <div style={{ textAlign: "center", padding: 60, color: "var(--color-text-dim)" }}>
          Chargement...
        </div>
      ) : sessions.length === 0 ? (
        <div
          className="card"
          style={{ textAlign: "center", padding: 60 }}
        >
          <div style={{ fontSize: 48, marginBottom: 16 }}>🤿</div>
          <h2 style={{ fontSize: 20, fontWeight: 600, marginBottom: 8 }}>Aucune session</h2>
          <p style={{ color: "var(--color-text-dim)", marginBottom: 20 }}>
            Créez votre première session d'annotation pour commencer
          </p>
          <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
            Créer une session
          </button>
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(380px, 1fr))", gap: 16 }}>
          {sessions.map((s, idx) => {
            const stats = getStats(s);
            return (
              <div
                key={s.name}
                className="card animate-fade-in"
                style={{
                  cursor: "pointer",
                  animationDelay: `${idx * 60}ms`,
                  animationFillMode: "backwards",
                }}
                onClick={() => navigate(`/session/${s.name}`)}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
                  <div>
                    <h3 style={{ fontSize: 17, fontWeight: 600, marginBottom: 4 }}>{s.name}</h3>
                    {s.description && (
                      <p style={{ fontSize: 13, color: "var(--color-text-dim)" }}>{s.description}</p>
                    )}
                  </div>
                  <button
                    className="btn btn-red"
                    style={{ padding: "4px 10px", fontSize: 12 }}
                    onClick={(e) => { e.stopPropagation(); setDeleteTarget(s.name); }}
                  >
                    Supprimer
                  </button>
                </div>

                <div style={{ fontSize: 12, color: "var(--color-text-dim)", marginBottom: 12 }}>
                  Créée le {new Date(s.created_at).toLocaleDateString("fr-FR", { day: "numeric", month: "long", year: "numeric" })}
                </div>

                <div className="progress-bar" style={{ marginBottom: 12 }}>
                  <div className="progress-fill" style={{ width: `${stats.pct}%` }} />
                </div>

                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <span className="badge badge-cyan">{stats.total} images</span>
                  <span className="badge badge-green">{stats.annotated} annotées</span>
                  {stats.ignored > 0 && <span className="badge badge-orange">{stats.ignored} ignorées</span>}
                  <span className="badge badge-gray" style={{ marginLeft: "auto" }}>{stats.pct}%</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Create Modal */}
      {showCreate && (
        <div className="modal-backdrop" onClick={() => setShowCreate(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: 20, fontWeight: 600, marginBottom: 20 }}>Nouvelle session</h2>

            <div style={{ marginBottom: 16 }}>
              <label style={{ display: "block", fontSize: 13, fontWeight: 500, marginBottom: 6, color: "var(--color-text-dim)" }}>
                Nom de la session *
              </label>
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Ex: essai_courant_5cms"
                autoFocus
              />
              {sanitized && sanitized !== newName && (
                <div style={{ fontSize: 12, color: "var(--color-accent)", marginTop: 6 }}>
                  → Nom sanitisé : <strong>{sanitized}</strong>
                </div>
              )}
              {sanitized && sanitized === newName && (
                <div style={{ fontSize: 12, color: "var(--color-green)", marginTop: 6 }}>
                  ✓ Nom valide
                </div>
              )}
            </div>

            <div style={{ marginBottom: 24 }}>
              <label style={{ display: "block", fontSize: 13, fontWeight: 500, marginBottom: 6, color: "var(--color-text-dim)" }}>
                Description (optionnelle)
              </label>
              <textarea
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="Conditions expérimentales, objectif..."
                rows={3}
              />
            </div>

            <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setShowCreate(false)}>Annuler</button>
              <button
                className="btn btn-primary"
                onClick={handleCreate}
                disabled={!sanitized || creating}
              >
                {creating ? "Création..." : "Créer la session"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="modal-backdrop" onClick={() => setDeleteTarget(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <h2 style={{ fontSize: 20, fontWeight: 600, marginBottom: 12, color: "var(--color-red)" }}>
              Supprimer la session ?
            </h2>
            <p style={{ color: "var(--color-text-dim)", marginBottom: 24 }}>
              La session <strong style={{ color: "var(--color-text)" }}>{deleteTarget}</strong> et toutes ses données
              (images, annotations, labels) seront supprimées définitivement.
            </p>
            <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
              <button className="btn btn-secondary" onClick={() => setDeleteTarget(null)}>Annuler</button>
              <button className="btn btn-red" onClick={handleDelete}>Supprimer définitivement</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
