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
  bg: "#0D1117", surface: "#161B22", surface2: "#21262D",
  accent: "#00FFFF", green: "#00FF88", orange: "#FFA500",
  red: "#FF4444", blue: "#3B82F6", border: "#30363D",
  text: "#E6EDF3", muted: "#8B949E",
};

interface AIPanelProps {
  sessionName: string;
  isGlobal?: boolean;
  preselectedSessions?: string;
}

function linReg(pts: { x: number; y: number }[]) {
  const n = pts.length;
  if (n < 2) return { a: 0, b: 0, r2: 0 };
  const sx = pts.reduce((s, p) => s + p.x, 0);
  const sy = pts.reduce((s, p) => s + p.y, 0);
  const sxx = pts.reduce((s, p) => s + p.x * p.x, 0);
  const sxy = pts.reduce((s, p) => s + p.x * p.y, 0);
  const a = (n * sxy - sx * sy) / (n * sxx - sx * sx || 1);
  const b = (sy - a * sx) / n;
  const yMean = sy / n;
  const ssTot = pts.reduce((s, p) => s + (p.y - yMean) ** 2, 0);
  const ssRes = pts.reduce((s, p) => s + (p.y - (a * p.x + b)) ** 2, 0);
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  return { a, b, r2 };
}

function regressionLine(a: number, b: number, xMin: number, xMax: number, n = 50) {
  return Array.from({ length: n }, (_, i) => {
    const x = xMin + (xMax - xMin) * i / (n - 1);
    return { x: parseFloat(x.toFixed(3)), y: parseFloat((a * x + b).toFixed(3)) };
  });
}

export default function AIPanel({ sessionName, isGlobal = false, preselectedSessions }: AIPanelProps) {
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
  const [globalModelName, setGlobalModelName] = useState("global_vc_model");
  const [angleVcData, setAngleVcData] = useState<{ theta: number; vc: number }[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [predictImageUrl, setPredictImageUrl] = useState<string | null>(null);

  const loadModels = useCallback(() => {
    listModels().then(setModels).catch(() => {});
  }, []);

  const loadAngleVc = useCallback(() => {
    const url = isGlobal
      ? "/api/train/global/angle-vc-data"
      : `/api/train/${encodeURIComponent(sessionName)}/angle-vc-data`;
    fetch(url)
      .then(r => r.ok ? r.json() : [])
      .then(d => setAngleVcData(d))
      .catch(() => {});
  }, [sessionName, isGlobal]);

  useEffect(() => { loadModels(); loadAngleVc(); }, [loadModels, loadAngleVc]);

  useEffect(() => {
    if (models.length > 0 && !selectedModel) setSelectedModel(models[0].name);
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
          loadAngleVc();
        }
      } catch {
        clearInterval(pollRef.current!);
        setIsTraining(false);
      }
    }, 500);
  }, [sessionName, isGlobal, loadModels, loadAngleVc]);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const handleTrain = async () => {
    setIsTraining(true);
    setProgress({ epoch: 0, total_epochs: epochs, status: "starting" });
    try {
      if (isGlobal) {
        await trainGlobalModel({
          sessions: preselectedSessions || undefined,
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
    ? Math.round((progress.epoch / progress.total_epochs) * 100) : 0;

  const lossData = progress?.train_losses?.map((tl, i) => ({
    epoch: i + 1,
    "Train Loss": +tl.toFixed(4),
    "Val Loss": progress.val_losses?.[i] !== undefined ? +(progress.val_losses[i]).toFixed(4) : undefined,
  })) ?? [];

  // ── Scatter Vc prédit vs réel — axes numériques, même domaine ─────────────────
  const scatterData = progress?.preds?.map((p, i) => ({
    x: parseFloat((progress.true?.[i] ?? 0).toFixed(2)),
    y: parseFloat(p.toFixed(2)),
  })) ?? [];

  const allVals = scatterData.flatMap(d => [d.x, d.y]);
  const rawMin = allVals.length > 0 ? Math.min(...allVals) : 0;
  const rawMax = allVals.length > 0 ? Math.max(...allVals) : 40;
  const mg = (rawMax - rawMin) * 0.12 + 1;
  const axMin = parseFloat((rawMin - mg).toFixed(1));
  const axMax = parseFloat((rawMax + mg).toFixed(1));
  const diagPts = [{ x: axMin, y: axMin }, { x: axMax, y: axMax }];

  // ── Vc = f(θ) ────────────────────────────────────────────────────────────────
  const annotPts = angleVcData.map(d => ({ x: d.theta, y: d.vc }));
  const regAnnot = linReg(annotPts);

  // Points prédits par le modèle : on associe theta_i aux preds du dernier entraînement
  const modelPredPts: { x: number; y: number }[] = [];
  if (progress?.preds && progress.preds.length > 0 && angleVcData.length > 0) {
    const nPreds = progress.preds.length;
    angleVcData.slice(-nPreds).forEach((d, i) => {
      modelPredPts.push({ x: d.theta, y: parseFloat((progress.preds![i]).toFixed(2)) });
    });
  }
  const regModel = linReg(modelPredPts);

  const allTh = [...annotPts, ...modelPredPts].map(p => p.x);
  const allVc = [...annotPts, ...modelPredPts].map(p => p.y);
  const thMin = allTh.length > 0 ? parseFloat((Math.min(...allTh) - 1).toFixed(1)) : 0;
  const thMax = allTh.length > 0 ? parseFloat((Math.max(...allTh) + 1).toFixed(1)) : 90;
  const vcMin = allVc.length > 0 ? parseFloat((Math.min(...allVc) - 2).toFixed(1)) : 0;
  const vcMax = allVc.length > 0 ? parseFloat((Math.max(...allVc) + 2).toFixed(1)) : 40;

  const regAnnotLine = annotPts.length >= 2 ? regressionLine(regAnnot.a, regAnnot.b, thMin, thMax) : [];
  const regModelLine = modelPredPts.length >= 2 ? regressionLine(regModel.a, regModel.b, thMin, thMax) : [];

  const TS = { background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 11 };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* HERO */}
      {!isGlobal && (
        <div style={{
          background: "linear-gradient(135deg, rgba(0,255,255,0.06) 0%, rgba(59,130,246,0.06) 100%)",
          border: `1px solid rgba(0,255,255,0.2)`, borderRadius: 16, padding: 24,
          display: "flex", alignItems: "center", gap: 20,
        }}>
          <div style={{ fontSize: 48, lineHeight: 1 }}>🌊</div>
          <div>
            <div style={{ fontSize: 20, fontWeight: 700, color: C.accent }}>Prédiction de Vc</div>
            <div style={{ fontSize: 13, color: C.muted, marginTop: 4, maxWidth: 420 }}>
              Entraîne un ResNet18 pour estimer la{" "}
              <span style={{ color: C.text, fontWeight: 600 }}>vitesse du courant marin (cm/s)</span>{" "}
              depuis une image de câble imergé.
            </div>
          </div>
        </div>
      )}

      {/* SECTION ENTRAÎNEMENT */}
      <Section icon="🧠" title="Entraîner un modèle Vc">
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 16, alignItems: "flex-end", marginBottom: isGlobal ? 14 : 0 }}>
          <div>
            <label style={labelStyle}>Époques</label>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input type="number" value={epochs} min={5} max={300}
                onChange={e => setEpochs(parseInt(e.target.value) || 50)}
                style={{ width: 80 }} />
              <span style={{ fontSize: 11, color: C.muted }}>(50 recommandé)</span>
            </div>
          </div>
          <button className="btn btn-primary" onClick={handleTrain} disabled={isTraining}
            style={{ height: 36, paddingLeft: 18, paddingRight: 18 }}>
            {isTraining ? "⏳ En cours..." : "🧠 Lancer l'entraînement"}
          </button>
        </div>

        {isGlobal && (
          <div style={{ marginBottom: 12 }}>
            <label style={labelStyle}>Nom du modèle</label>
            <input type="text" value={globalModelName} onChange={e => setGlobalModelName(e.target.value)}
              placeholder="global_vc_model" style={{ width: 260 }} />
            {preselectedSessions && (
              <div style={{ fontSize: 11, color: C.muted, marginTop: 6 }}>
                Sessions : <span style={{ color: C.accent }}>{preselectedSessions}</span>
              </div>
            )}
          </div>
        )}

        {/* Progression */}
        {progress && progress.status !== "idle" && (
          <div style={{
            background: C.surface2, borderRadius: 12, padding: 18,
            border: `1px solid ${C.border}`, marginTop: 14,
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
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
              <div style={{ background: C.border, borderRadius: 4, height: 6, marginBottom: 12, overflow: "hidden" }}>
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
                {/* Metriques */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
                  <MCard label="MAE" value={`${progress.mae} cm/s`} color={C.accent} />
                  <MCard label="RMSE" value={`${progress.rmse} cm/s`} color={C.blue} />
                  <MCard label="Dataset" value={`${progress.n_train}+${progress.n_val}`} color={C.green} sub="train+val" />
                </div>

                {/* Courbe de loss */}
                {lossData.length > 1 && (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: C.muted, marginBottom: 8 }}>Courbe de loss</div>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={lossData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="epoch" tick={{ fontSize: 9, fill: C.muted }} />
                        <YAxis tick={{ fontSize: 9, fill: C.muted }} />
                        <Tooltip contentStyle={TS} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Line type="monotone" dataKey="Train Loss" stroke={C.accent} dot={false} strokeWidth={2} />
                        <Line type="monotone" dataKey="Val Loss" stroke={C.orange} dot={false} strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Scatter Vc prédit vs réel — ORTHONORMÉ */}
                {scatterData.length > 0 && (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: C.muted, marginBottom: 4 }}>Vc prédit vs réel (cm/s)</div>
                    <div style={{ fontSize: 10, color: C.muted, marginBottom: 8 }}>
                      Chaque point = une image de validation · droite pointillée = prédiction parfaite (y = x)
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                      <ScatterChart margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis
                          dataKey="x"
                          type="number"
                          name="Vc réel"
                          domain={[axMin, axMax]}
                          tickCount={6}
                          label={{ value: "Vc réel (cm/s)", position: "insideBottom", offset: -14, fontSize: 10, fill: C.muted }}
                          tick={{ fontSize: 9, fill: C.muted }}
                          axisLine={{ stroke: C.border }}
                          tickLine={false}
                        />
                        <YAxis
                          dataKey="y"
                          type="number"
                          name="Vc prédit"
                          domain={[axMin, axMax]}
                          tickCount={6}
                          label={{ value: "Vc prédit (cm/s)", angle: -90, position: "insideLeft", offset: 14, fontSize: 10, fill: C.muted }}
                          tick={{ fontSize: 9, fill: C.muted }}
                          axisLine={{ stroke: C.border }}
                          tickLine={false}
                        />
                        <Tooltip contentStyle={TS} formatter={(v: any) => [`${v} cm/s`]} />
                        {/* Diagonale y=x */}
                        <Scatter
                          data={diagPts}
                          line={{ stroke: "#888", strokeDasharray: "5 3", strokeWidth: 1.5 }}
                          shape={() => null as any}
                          legendType="none"
                          name="y=x"
                        />
                        {/* Mesures */}
                        <Scatter data={scatterData} fill={C.green} opacity={0.85} name="Mesures" />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Graphique Vc = f(θ) */}
                {annotPts.length >= 3 && (
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: C.muted, marginBottom: 4 }}>Vc = f(θ) — angle PCA du câble</div>
                    <div style={{ fontSize: 10, color: C.muted, marginBottom: 2 }}>
                      <span style={{ color: C.accent }}>● Annotations</span>
                      {" : Vc = "}{regAnnot.a.toFixed(3)}·θ {regAnnot.b >= 0 ? "+" : ""}{regAnnot.b.toFixed(2)} cm/s
                      <span style={{ color: C.muted, marginLeft: 6 }}>(R² = {regAnnot.r2.toFixed(3)})</span>
                    </div>
                    {regModelLine.length > 0 && (
                      <div style={{ fontSize: 10, color: C.muted, marginBottom: 8 }}>
                        <span style={{ color: C.green }}>● Modèle NN</span>
                        {" : Vc = "}{regModel.a.toFixed(3)}·θ {regModel.b >= 0 ? "+" : ""}{regModel.b.toFixed(2)} cm/s
                        <span style={{ color: C.muted, marginLeft: 6 }}>(R² = {regModel.r2.toFixed(3)})</span>
                      </div>
                    )}
                    <ResponsiveContainer width="100%" height={240}>
                      <ScatterChart margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis
                          dataKey="x"
                          type="number"
                          name="θ"
                          domain={[thMin, thMax]}
                          tickCount={7}
                          label={{ value: "θ (°)", position: "insideBottom", offset: -14, fontSize: 10, fill: C.muted }}
                          tick={{ fontSize: 9, fill: C.muted }}
                          axisLine={{ stroke: C.border }}
                          tickLine={false}
                        />
                        <YAxis
                          dataKey="y"
                          type="number"
                          name="Vc"
                          domain={[vcMin, vcMax]}
                          tickCount={6}
                          label={{ value: "Vc (cm/s)", angle: -90, position: "insideLeft", offset: 14, fontSize: 10, fill: C.muted }}
                          tick={{ fontSize: 9, fill: C.muted }}
                          axisLine={{ stroke: C.border }}
                          tickLine={false}
                        />
                        <Tooltip contentStyle={TS}
                          formatter={(v: any, name: string) =>
                            name === "θ" ? [`${v}°`, "θ"] : [`${v} cm/s`, "Vc"]
                          }
                        />
                        <Legend wrapperStyle={{ fontSize: 10 }} />
                        {/* Nuage annotations */}
                        <Scatter data={annotPts} fill={C.accent} opacity={0.6} name="Annotations" />
                        {/* Régression annotations */}
                        {regAnnotLine.length > 0 && (
                          <Scatter
                            data={regAnnotLine}
                            line={{ stroke: C.accent, strokeWidth: 1.5 }}
                            shape={() => null as any}
                            legendType="none"
                            name="rég. annot."
                          />
                        )}
                        {/* Points prédits par le modèle NN */}
                        {modelPredPts.length > 0 && (
                          <Scatter data={modelPredPts} fill={C.green} opacity={0.8} name="Modèle NN" />
                        )}
                        {/* Droite du modèle NN */}
                        {regModelLine.length > 0 && (
                          <Scatter
                            data={regModelLine}
                            line={{ stroke: C.green, strokeWidth: 2, strokeDasharray: "6 3" }}
                            shape={() => null as any}
                            legendType="none"
                            name="rég. modèle"
                          />
                        )}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </Section>

      {/* SECTION PRÉDICTION */}
      <Section icon="🔍" title="Prédire Vc sur une image">
        {models.length === 0 ? (
          <div style={{
            padding: 24, textAlign: "center", color: C.muted, fontSize: 13,
            background: C.surface2, borderRadius: 12, border: `1px dashed ${C.border}`,
          }}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>🤖</div>
            Aucun modèle disponible.<br />
            <span style={{ color: C.text }}>Annotez des images avec <code style={{ color: C.accent }}>current_speed_cm_s</code> puis cliquez Entraîner.</span>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div>
                <label style={labelStyle}>Modèle</label>
                <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)} style={{ width: "100%" }}>
                  {models.map(m => (
                    <option key={m.name} value={m.name}>
                      {m.is_global ? "🌍 " : "📂 "}{m.name} ({m.size_mb} MB)
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label style={labelStyle}>Image à analyser</label>
                <div className={`drop-zone ${dragOver ? "drag-over" : ""}`} style={{ minHeight: 90, cursor: "pointer" }}
                  onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handlePredictFile(f); }}
                  onClick={() => fileInputRef.current?.click()}>
                  <input ref={fileInputRef} type="file" accept="image/*" hidden
                    onChange={e => { const f = e.target.files?.[0]; if (f) handlePredictFile(f); }} />
                  {predictImageUrl ? (
                    <img src={predictImageUrl} alt="preview"
                      style={{ maxWidth: "100%", maxHeight: 120, borderRadius: 8, objectFit: "contain" }} />
                  ) : (
                    <><div style={{ fontSize: 24, marginBottom: 4 }}>📷</div><div style={{ fontSize: 12, color: C.muted }}>Glissez ou cliquez</div></>
                  )}
                </div>
              </div>
              <button className="btn btn-primary" onClick={handlePredict}
                disabled={!predictFile || !selectedModel || isPredicting}
                style={{ justifyContent: "center" }}>
                {isPredicting ? "⏳ Analyse..." : "🔍 Analyser"}
              </button>
            </div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center" }}>
              {predictResult !== null ? (
                <div style={{
                  background: "linear-gradient(135deg, rgba(0,255,255,0.1), rgba(0,255,136,0.1))",
                  border: `2px solid ${C.accent}`, borderRadius: 16, padding: 28,
                  textAlign: "center", width: "100%",
                }}>
                  <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: 1, marginBottom: 6 }}>Vitesse estimée</div>
                  <div style={{ fontSize: 48, fontWeight: 900, color: C.accent, lineHeight: 1, marginBottom: 6 }}>{predictResult}</div>
                  <div style={{ fontSize: 18, color: C.text, fontWeight: 500 }}>cm/s</div>
                  <div style={{ marginTop: 12, fontSize: 11, color: C.muted }}>Modèle : <span style={{ color: C.text }}>{selectedModel}</span></div>
                </div>
              ) : predictError ? (
                <div style={{
                  background: "rgba(255,68,68,0.08)", border: `1px solid rgba(255,68,68,0.3)`,
                  borderRadius: 12, padding: 20, textAlign: "center", color: C.red, fontSize: 13, width: "100%",
                }}>❌ {predictError}</div>
              ) : (
                <div style={{ textAlign: "center", color: C.muted, fontSize: 13 }}>
                  <div style={{ fontSize: 36, marginBottom: 10, opacity: 0.25 }}>🌊</div>
                  Résultat ici
                </div>
              )}
            </div>
          </div>
        )}
      </Section>

      {/* MODÈLES + TÉLÉCHARGEMENT */}
      {!isGlobal && models.length > 0 && (
        <Section icon="📦" title="Modèles entraînés">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {models.map(m => (
              <div key={m.name} style={{
                display: "flex", justifyContent: "space-between", alignItems: "center",
                padding: "10px 14px", background: C.bg, borderRadius: 10, border: `1px solid ${C.border}`,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ fontSize: 16 }}>{m.is_global ? "🌍" : "📂"}</span>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{m.name}</div>
                    <div style={{ fontSize: 10, color: C.muted }}>{m.size_mb} MB — {new Date(m.modified_at).toLocaleDateString("fr-FR")}</div>
                  </div>
                </div>
                <a
                  href={`http://localhost:8000/api/models/${encodeURIComponent(m.filename)}/download`}
                  className="btn btn-secondary"
                  style={{ fontSize: 11, textDecoration: "none", padding: "4px 12px" }}
                  download={m.filename}
                >
                  ⬇ Télécharger
                </a>
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({ icon, title, children }: { icon: string; title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "#161B22", border: "1px solid #30363D", borderRadius: 14, overflow: "hidden" }}>
      <div style={{
        padding: "12px 20px", borderBottom: "1px solid #30363D",
        display: "flex", alignItems: "center", gap: 8,
        background: "rgba(255,255,255,0.02)",
      }}>
        <span style={{ fontSize: 16 }}>{icon}</span>
        <span style={{ fontWeight: 700, fontSize: 14, color: "#E6EDF3" }}>{title}</span>
      </div>
      <div style={{ padding: 20 }}>{children}</div>
    </div>
  );
}

function MCard({ label, value, color, sub }: { label: string; value: string; color: string; sub?: string }) {
  return (
    <div style={{ background: "#0D1117", borderRadius: 8, padding: "12px 14px", border: "1px solid #30363D", textAlign: "center" }}>
      <div style={{ fontSize: 19, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 11, color: "#8B949E", marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 10, color: "#8B949E" }}>{sub}</div>}
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: "block", fontSize: 12, color: "#8B949E",
  marginBottom: 5, fontWeight: 500,
};
