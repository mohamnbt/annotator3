import { useState, useEffect, useRef, useCallback } from "react";
import {
  trainSessionModel, getTrainProgress, trainGlobalModel, getGlobalTrainProgress,
  listModels, predictVc,
  type TrainProgress, type ModelMeta,
} from "../lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ScatterChart, Scatter, ResponsiveContainer,
} from "recharts";

const C = {
  bg: "#0D1117",
  surface: "#161B22",
  surface2: "#21262D",
  accent: "#00FFFF",
  green: "#00FF88",
  orange: "#FFA500",
  red: "#FF4444",
  blue: "#3B82F6",
  border: "#30363D",
  text: "#E6EDF3",
  muted: "#8B949E",
  purple: "#a78bfa",
};

interface AIPanelProps {
  sessionName: string;
  isGlobal?: boolean;
}

export default function AIPanel({ sessionName, isGlobal = false }: AIPanelProps) {
  const [epochs, setEpochs] = useState(50);
  const [progress, setProgress] = useState<TrainProgress | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predictResult, setPredictResult] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [predictImageUrl, setPredictImageUrl] = useState<string | null>(null);
  const [globalSessions, setGlobalSessions] = useState("");
  const [globalModelName, setGlobalModelName] = useState("global_vc_model");

  const loadModels = useCallback(() => {
    listModels().then(setModels).catch(() => {});
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      setSelectedModel(models[0].name);
    }
  }, [models, selectedModel]);

  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const p = isGlobal
          ? await getGlobalTrainProgress()
          : await getTrainProgress(sessionName);
        setProgress(p);
        if (p.status === "done" || p.status === "error") {
          clearInterval(pollRef.current!);
          setIsTraining(false);
          loadModels();
        }
      } catch {
        clearInterval(pollRef.current!);
        setIsTraining(false);
      }
    }, 500);
  }, [sessionName, isGlobal, loadModels]);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const handleTrain = async () => {
    setIsTraining(true);
    setProgress({ epoch: 0, total_epochs: epochs, status: "starting" });
    try {
      if (isGlobal) {
        await trainGlobalModel({
          sessions: globalSessions || undefined,
          epochs,
          model_name: globalModelName,
        });
      } else {
        await trainSessionModel(sessionName, epochs);
      }
      startPolling();
    } catch (e: any) {
      setProgress({ epoch: 0, total_epochs: epochs, status: "error", error: e.message });
      setIsTraining(false);
    }
  };

  const handlePredictFile = (file: File) => {
    setPredictFile(file);
    setPredictResult(null);
    setPredictError(null);
    const url = URL.createObjectURL(file);
    if (predictImageUrl) URL.revokeObjectURL(predictImageUrl);
    setPredictImageUrl(url);
  };

  const handlePredict = async () => {
    if (!predictFile || !selectedModel) return;
    setIsPredicting(true);
    setPredictResult(null);
    setPredictError(null);
    try {
      const r = await predictVc(selectedModel, predictFile);
      setPredictResult(r.vitesse_estimee);
    } catch (e: any) {
      setPredictError(e.message);
    } finally {
      setIsPredicting(false);
    }
  };

  const pct = progress && progress.total_epochs > 0
    ? Math.round((progress.epoch / progress.total_epochs) * 100)
    : 0;

  const lossData = progress?.train_losses?.map((tl, i) => ({
    epoch: i + 1,
    "Train Loss": +tl.toFixed(4),
    "Val Loss": progress.val_losses?.[i] !== undefined ? +(progress.val_losses[i]).toFixed(4) : undefined,
  })) ?? [];

  const scatterData = progress?.preds?.map((p, i) => ({
    x: progress.true?.[i] ?? 0,
    y: p,
  })) ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* ── HERO ── */}
      <div style={{
        background: "linear-gradient(135deg, rgba(0,255,255,0.06) 0%, rgba(59,130,246,0.06) 100%)",
        border: `1px solid rgba(0,255,255,0.2)`,
        borderRadius: 16, padding: 28,
        display: "flex", alignItems: "center", gap: 24,
      }}>
        <div style={{ fontSize: 52, lineHeight: 1 }}>🌊</div>
        <div>
          <div style={{ fontSize: 22, fontWeight: 700, color: C.accent, letterSpacing: "-0.5px" }}>
            Prédiction de Vc
          </div>
          <div style={{ fontSize: 14, color: C.muted, marginTop: 4, maxWidth: 460 }}>
            Entraîne un ResNet18 sur vos annotations pour estimer la{" "}
            <span style={{ color: C.text, fontWeight: 600 }}>vitesse du courant marin (cm/s)</span>{" "}
            directement depuis une image de câble immergé.
          </div>
        </div>
      </div>

      {/* ── SECTION ENTRAÎNEMENT ── */}
      <Section icon="🧠" title="Entraîner un modèle Vc">
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 16, alignItems: "flex-end", marginBottom: 16 }}>
          <div>
            <label style={labelStyle}>Époques d'entraînement</label>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="number"
                value={epochs}
                min={5}
                max={300}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 50)}
                style={{ width: 90 }}
              />
              <span style={{ fontSize: 12, color: C.muted }}>
                (50 recommandé pour un premier test)
              </span>
            </div>
          </div>
          <button
            className="btn btn-primary"
            onClick={handleTrain}
            disabled={isTraining}
            style={{ height: 38, paddingLeft: 20, paddingRight: 20, whiteSpace: "nowrap" }}
          >
            {isTraining ? "⏳ Entraînement..." : "🧠 Lancer l'entraînement"}
          </button>
        </div>

        {isGlobal && (
          <div style={{ marginBottom: 16 }}>
            <label style={labelStyle}>Sessions à inclure (vide = toutes)</label>
            <input
              type="text"
              value={globalSessions}
              onChange={(e) => setGlobalSessions(e.target.value)}
              placeholder="ex: session1,session2 — vide = toutes"
              style={{ width: "100%" }}
            />
            <div style={{ marginTop: 12 }}>
              <label style={labelStyle}>Nom du modèle</label>
              <input
                type="text"
                value={globalModelName}
                onChange={(e) => setGlobalModelName(e.target.value)}
                placeholder="global_vc_model"
                style={{ width: 240 }}
              />
            </div>
          </div>
        )}

        {/* Progression */}
        {progress && progress.status !== "idle" && (
          <div style={{
            background: C.surface2, borderRadius: 12, padding: 20,
            border: `1px solid ${C.border}`, marginTop: 4,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: C.text }}>
                {progress.status === "starting" && "⏳ Démarrage..."}
                {progress.status === "running" && `Époque ${progress.epoch} / ${progress.total_epochs}`}
                {progress.status === "done" && "✅ Entraînement terminé"}
                {progress.status === "error" && "❌ Erreur"}
              </span>
              {progress.status === "running" && (
                <span style={{ fontSize: 13, color: C.accent, fontWeight: 700 }}>{pct}%</span>
              )}
            </div>

            {(progress.status === "running" || progress.status === "done") && (
              <div style={{ background: C.border, borderRadius: 4, height: 6, marginBottom: 14, overflow: "hidden" }}>
                <div style={{
                  height: "100%", borderRadius: 4,
                  width: `${progress.status === "done" ? 100 : pct}%`,
                  background: progress.status === "done"
                    ? `linear-gradient(90deg, ${C.green}, ${C.accent})`
                    : `linear-gradient(90deg, ${C.accent}, ${C.blue})`,
                  transition: "width 0.4s ease",
                }} />
              </div>
            )}

            {progress.status === "error" && (
              <div style={{ color: C.red, fontSize: 13, padding: "8px 12px", background: "rgba(255,68,68,0.08)", borderRadius: 8 }}>
                {progress.error}
              </div>
            )}

            {progress.status === "done" && progress.mae !== undefined && (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 20 }}>
                  <MetricCard label="MAE" value={`${progress.mae} cm/s`} color={C.accent} />
                  <MetricCard label="RMSE" value={`${progress.rmse} cm/s`} color={C.blue} />
                  <MetricCard
                    label="Dataset"
                    value={`${progress.n_train}+${progress.n_val}`}
                    color={C.green}
                    sub="train + val"
                  />
                </div>

                {lossData.length > 1 && (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, color: C.muted, marginBottom: 10 }}>Courbe de loss</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={lossData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: C.muted }} />
                        <YAxis tick={{ fontSize: 10, fill: C.muted }} />
                        <Tooltip contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
                        <Legend wrapperStyle={{ fontSize: 12, color: C.muted }} />
                        <Line type="monotone" dataKey="Train Loss" stroke={C.accent} dot={false} strokeWidth={2} />
                        <Line type="monotone" dataKey="Val Loss" stroke={C.orange} dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {scatterData.length > 0 && (
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: C.muted, marginBottom: 10 }}>Vc prédit vs réel (cm/s)</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="x" name="Vc réel" tick={{ fontSize: 10, fill: C.muted }} label={{ value: "Réel", position: "insideBottom", offset: -2, fontSize: 10, fill: C.muted }} />
                        <YAxis dataKey="y" name="Vc prédit" tick={{ fontSize: 10, fill: C.muted }} label={{ value: "Prédit", angle: -90, position: "insideLeft", fontSize: 10, fill: C.muted }} />
                        <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, fontSize: 12 }} />
                        <Scatter data={scatterData} fill={C.green} opacity={0.8} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </Section>

      {/* ── SECTION PRÉDICTION ── */}
      <Section icon="🔍" title="Prédire Vc sur une image">
        {models.length === 0 ? (
          <div style={{
            padding: 24, textAlign: "center", color: C.muted, fontSize: 13,
            background: C.surface2, borderRadius: 12, border: `1px dashed ${C.border}`,
          }}>
            <div style={{ fontSize: 32, marginBottom: 8 }}>🤖</div>
            Aucun modèle disponible.<br />
            <span style={{ color: C.text }}>Annotez des images avec <code style={{ color: C.accent }}>current_speed_cm_s</code> puis cliquez Entraîner.</span>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            {/* Colonne gauche : upload + modèle */}
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              <div>
                <label style={labelStyle}>Modèle à utiliser</label>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  style={{ width: "100%" }}
                >
                  {models.map((m) => (
                    <option key={m.name} value={m.name}>
                      {m.is_global ? "🌍 " : "📂 "}{m.name} ({m.size_mb} MB)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label style={labelStyle}>Image à analyser</label>
                <div
                  className={`drop-zone ${dragOver ? "drag-over" : ""}`}
                  style={{ minHeight: 100, cursor: "pointer" }}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault(); setDragOver(false);
                    const f = e.dataTransfer.files[0];
                    if (f) handlePredictFile(f);
                  }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    hidden
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) handlePredictFile(f); }}
                  />
                  {predictImageUrl ? (
                    <img
                      src={predictImageUrl}
                      alt="prévisualisation"
                      style={{ maxWidth: "100%", maxHeight: 140, borderRadius: 8, objectFit: "contain" }}
                    />
                  ) : (
                    <>
                      <div style={{ fontSize: 28, marginBottom: 6 }}>📷</div>
                      <div style={{ fontSize: 13, color: C.muted }}>Glissez une image ici ou cliquez</div>
                    </>
                  )}
                </div>
              </div>

              <button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={!predictFile || !selectedModel || isPredicting}
                style={{ justifyContent: "center" }}
              >
                {isPredicting ? "⏳ Analyse en cours..." : "🔍 Analyser"}
              </button>
            </div>

            {/* Colonne droite : résultat */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
              {predictResult !== null ? (
                <div style={{
                  background: "linear-gradient(135deg, rgba(0,255,255,0.1), rgba(0,255,136,0.1))",
                  border: `2px solid ${C.accent}`,
                  borderRadius: 16, padding: 32, textAlign: "center", width: "100%",
                }}>
                  <div style={{ fontSize: 13, color: C.muted, marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>
                    Vitesse estimée
                  </div>
                  <div style={{ fontSize: 52, fontWeight: 800, color: C.accent, lineHeight: 1, marginBottom: 8 }}>
                    {predictResult}
                  </div>
                  <div style={{ fontSize: 20, color: C.text, fontWeight: 500 }}>cm/s</div>
                  <div style={{ marginTop: 16, fontSize: 12, color: C.muted }}>
                    Modèle : <span style={{ color: C.text }}>{selectedModel}</span>
                  </div>
                </div>
              ) : predictError ? (
                <div style={{
                  background: "rgba(255,68,68,0.08)", border: `1px solid rgba(255,68,68,0.3)`,
                  borderRadius: 12, padding: 20, textAlign: "center", color: C.red, fontSize: 13, width: "100%",
                }}>
                  ❌ {predictError}
                </div>
              ) : (
                <div style={{ textAlign: "center", color: C.muted, fontSize: 13 }}>
                  <div style={{ fontSize: 40, marginBottom: 12, opacity: 0.3 }}>🌊</div>
                  Le résultat apparaîtra ici
                </div>
              )}
            </div>
          </div>
        )}
      </Section>

      {/* ── MODÈLES DISPONIBLES ── */}
      {models.length > 0 && (
        <Section icon="📦" title="Modèles entraînés">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {models.map((m) => (
              <div key={m.name} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "12px 16px",
                background: C.surface2, borderRadius: 10, border: `1px solid ${C.border}`,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 18 }}>{m.is_global ? "🌍" : "📂"}</span>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 14, color: C.text }}>{m.name}</div>
                    <div style={{ fontSize: 11, color: C.muted }}>Modifié {new Date(m.modified_at).toLocaleDateString("fr-FR")}</div>
                  </div>
                </div>
                <span style={{ fontSize: 12, color: C.muted, background: C.surface, padding: "3px 10px", borderRadius: 20, border: `1px solid ${C.border}` }}>
                  {m.size_mb} MB
                </span>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

// ── Sous-composants ───────────────────────────────────────────────────────────

function Section({ icon, title, children }: { icon: string; title: string; children: React.ReactNode }) {
  return (
    <div style={{
      background: "#161B22",
      border: "1px solid #30363D",
      borderRadius: 14, overflow: "hidden",
    }}>
      <div style={{
        padding: "14px 20px", borderBottom: "1px solid #30363D",
        display: "flex", alignItems: "center", gap: 10,
        background: "rgba(255,255,255,0.02)",
      }}>
        <span style={{ fontSize: 18 }}>{icon}</span>
        <span style={{ fontWeight: 700, fontSize: 15, color: "#E6EDF3" }}>{title}</span>
      </div>
      <div style={{ padding: 20 }}>{children}</div>
    </div>
  );
}

function MetricCard({ label, value, color, sub }: { label: string; value: string; color: string; sub?: string }) {
  return (
    <div style={{
      background: "#0D1117", borderRadius: 10, padding: "14px 16px",
      border: "1px solid #30363D", textAlign: "center",
    }}>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 12, color: "#8B949E", marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: "#8B949E" }}>{sub}</div>}
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: "block", fontSize: 12, color: "#8B949E",
  marginBottom: 6, fontWeight: 500,
};
