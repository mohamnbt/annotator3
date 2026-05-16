import { useState, useEffect, useRef, useCallback } from "react";
import {
  trainSessionModel, getTrainProgress, trainGlobalModel, getGlobalTrainProgress,
  listModels, listSessions, predictVc, getAngleVcData,
  type TrainProgress, type ModelMeta, type SessionMeta,
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
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [predictFile, setPredictFile] = useState<File | null>(null);
  const [predictResult, setPredictResult] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [globalModelName, setGlobalModelName] = useState("global_vc_model");

  // Vc = f(θ) — sélecteur de source
  // "" = global (toutes sessions), ou le nom d'une session spécifique
  const [vcSource, setVcSource] = useState<string>(isGlobal ? "" : sessionName);
  const [angleVcData, setAngleVcData] = useState<{ theta: number; vc: number }[]>([]);
  const [vcLoading, setVcLoading] = useState(false);

  // Visualisation d'un modèle existant (courbe Vc_NN = f(θ))
  const [vizModel, setVizModel] = useState<string | null>(null);
  const [vizData, setVizData] = useState<{ theta: number; vc: number }[]>([]);
  const [vizLoading, setVizLoading] = useState(false);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [predictImageUrl, setPredictImageUrl] = useState<string | null>(null);

  const loadModels = useCallback(() => {
    listModels().then(setModels).catch(() => {});
  }, []);

  const loadSessions = useCallback(() => {
    listSessions().then(setSessions).catch(() => {});
  }, []);

  // Charge les données Vc=f(θ) selon la source choisie
  const loadAngleVc = useCallback((source: string) => {
    setVcLoading(true);
    const p = source === "" ? getAngleVcData() : getAngleVcData(source);
    p.then(d => { if (Array.isArray(d)) setAngleVcData(d); })
     .catch(() => {})
     .finally(() => setVcLoading(false));
  }, []);

  useEffect(() => {
    loadModels();
    loadSessions();
    loadAngleVc(vcSource);
  }, [loadModels, loadSessions, loadAngleVc, vcSource]);

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
        if (p.angle_vc_pts && Array.isArray(p.angle_vc_pts) && p.angle_vc_pts.length > 0) {
          setAngleVcData(p.angle_vc_pts);
        }
        if (p.status === "done" || p.status === "error") {
          clearInterval(pollRef.current!);
          setIsTraining(false);
          loadModels();
          loadAngleVc(vcSource);
        }
      } catch {
        clearInterval(pollRef.current!);
        setIsTraining(false);
      }
    }, 500);
  }, [sessionName, isGlobal, loadModels, loadAngleVc, vcSource]);

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

  // Visualise la courbe Vc_NN d'un modèle existant en faisant passer toutes les images annotées
  // On utilise ici les données réelles de la source sélectionnée + on ne peut pas avoir les
  // prédictions NN sans réentraîner (le backend ne stocke pas les preds après redémarrage).
  // On affiche donc : courbe réelle de la source + info du modèle.
  // Pour afficher les preds NN d'un modèle sauvegardé, il faudrait une route /api/models/{name}/predict-all.
  // Pour l'instant on montre : données réelles de la source choisie.
  const handleVisualize = async (modelName: string) => {
    if (vizModel === modelName) { setVizModel(null); setVizData([]); return; }
    setVizModel(modelName);
    setVizLoading(true);
    try {
      // Source = session du modèle si c'est un modèle par session, sinon global
      const isGlobalModel = modelName.startsWith("global") || !modelName.endsWith("_vc_model");
      const src = isGlobalModel ? undefined : modelName.replace(/_vc_model$/, "");
      const d = await getAngleVcData(src);
      if (Array.isArray(d)) setVizData(d);
    } catch {
      setVizData([]);
    } finally {
      setVizLoading(false);
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

  // ── Vc = f(θ) — source sélectionnée ───────────────────────────────────────
  // Pendant l'entraînement : on garde les angle_vc_pts injectés par le backend
  const rawAngleVc: { theta: number; vc: number }[] =
    (progress?.angle_vc_pts && progress.angle_vc_pts.length > 0)
      ? progress.angle_vc_pts
      : angleVcData;

  const annotPts = rawAngleVc.map(d => ({ x: d.theta, y: d.vc }));
  const regAnnot = linReg(annotPts);

  // Courbe NN (après entraînement, depuis nn_angle_vc_pts)
  const nnPts = (progress as any)?.nn_angle_vc_pts
    ? (progress as any).nn_angle_vc_pts.map((d: any) => ({ x: d.theta, y: d.vc }))
    : [];
  const regNN = linReg(nnPts);

  const allTh = [...annotPts, ...nnPts].map(p => p.x);
  const allVcY = [...annotPts, ...nnPts].map(p => p.y);
  const thMin = allTh.length > 0 ? parseFloat((Math.min(...allTh) - 1).toFixed(1)) : 0;
  const thMax = allTh.length > 0 ? parseFloat((Math.max(...allTh) + 1).toFixed(1)) : 90;
  const vcMin = allVcY.length > 0 ? parseFloat((Math.min(...allVcY) - 2).toFixed(1)) : 0;
  const vcMax = allVcY.length > 0 ? parseFloat((Math.max(...allVcY) + 2).toFixed(1)) : 40;

  const regAnnotLine = annotPts.length >= 2 ? regressionLine(regAnnot.a, regAnnot.b, thMin, thMax) : [];
  const regNNLine = nnPts.length >= 2 ? regressionLine(regNN.a, regNN.b, thMin, thMax) : [];

  const showVcTheta = annotPts.length >= 2;

  // ── Viz modèle existant ────────────────────────────────────────────────────
  const vizPts = vizData.map(d => ({ x: d.theta, y: d.vc }));
  const regViz = linReg(vizPts);
  const vizThMin = vizPts.length > 0 ? parseFloat((Math.min(...vizPts.map(p => p.x)) - 1).toFixed(1)) : 0;
  const vizThMax = vizPts.length > 0 ? parseFloat((Math.max(...vizPts.map(p => p.x)) + 1).toFixed(1)) : 90;
  const vizVcMin = vizPts.length > 0 ? parseFloat((Math.min(...vizPts.map(p => p.y)) - 2).toFixed(1)) : 0;
  const vizVcMax = vizPts.length > 0 ? parseFloat((Math.max(...vizPts.map(p => p.y)) + 2).toFixed(1)) : 40;
  const regVizLine = vizPts.length >= 2 ? regressionLine(regViz.a, regViz.b, vizThMin, vizThMax) : [];

  // Dossiers uniques pour le sélecteur
  const folders = Array.from(new Set(sessions.map(s => s.folder).filter(Boolean)));

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

      {/* Vc = f(θ) — avec sélecteur de source */}
      <Section icon="📐" title="Vc = f(θ) — angle PCA du câble">
        {/* Sélecteur de source */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14, flexWrap: "wrap" }}>
          <label style={{ ...labelStyle, marginBottom: 0, whiteSpace: "nowrap" }}>Source :</label>
          <select
            value={vcSource}
            onChange={e => { setVcSource(e.target.value); setProgress(null); }}
            style={{ flex: "0 0 auto", minWidth: 220, fontSize: 12 }}
          >
            <option value="">🌍 Toutes les sessions</option>
            {folders.length > 0 && (
              <optgroup label="── Par dossier ──" />
            )}
            {folders.map(f => (
              <optgroup key={f} label={`📁 ${f}`}>
                {sessions.filter(s => s.folder === f).map(s => (
                  <option key={s.name} value={s.name}>
                    📂 {s.name} ({s.images.filter(i => i.status === "annotated").length} ann.)
                  </option>
                ))}
              </optgroup>
            ))}
            {sessions.filter(s => !s.folder).map(s => (
              <option key={s.name} value={s.name}>
                📂 {s.name} ({s.images.filter(i => i.status === "annotated").length} ann.)
              </option>
            ))}
          </select>
          {vcLoading && <span style={{ fontSize: 11, color: C.muted }}>⏳ chargement…</span>}
        </div>

        {!showVcTheta ? (
          <div style={{
            padding: 24, textAlign: "center", color: C.muted, fontSize: 13,
            background: C.surface2, borderRadius: 12, border: `1px dashed ${C.border}`,
          }}>
            <div style={{ fontSize: 28, marginBottom: 8 }}>📐</div>
            Pas encore assez d'annotations avec θ et Vc pour cette source.
          </div>
        ) : (
          <>
            <div style={{ fontSize: 10, color: C.muted, marginBottom: 4 }}>
              <span style={{ color: C.accent }}>● Annotations</span>
              {" : Vc = "}{regAnnot.a.toFixed(3)}·θ {regAnnot.b >= 0 ? "+" : ""}{regAnnot.b.toFixed(2)} cm/s
              <span style={{ color: C.muted, marginLeft: 8 }}>R² = {regAnnot.r2.toFixed(3)}</span>
              <span style={{ color: C.muted, marginLeft: 8 }}>({annotPts.length} pts)</span>
            </div>
            {regNNLine.length > 0 && (
              <div style={{ fontSize: 10, color: C.muted, marginBottom: 8 }}>
                <span style={{ color: C.green }}>● Modèle NN</span>
                {" : Vc = "}{regNN.a.toFixed(3)}·θ {regNN.b >= 0 ? "+" : ""}{regNN.b.toFixed(2)} cm/s
                <span style={{ color: C.muted, marginLeft: 8 }}>R² = {regNN.r2.toFixed(3)}</span>
              </div>
            )}
            <ResponsiveContainer width="100%" height={260}>
              <ScatterChart margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis
                  dataKey="x" type="number" name="θ"
                  domain={[thMin, thMax]} tickCount={7}
                  label={{ value: "θ (°)", position: "insideBottom", offset: -14, fontSize: 10, fill: C.muted }}
                  tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.border }} tickLine={false}
                />
                <YAxis
                  dataKey="y" type="number" name="Vc"
                  domain={[vcMin, vcMax]} tickCount={6}
                  label={{ value: "Vc (cm/s)", angle: -90, position: "insideLeft", offset: 14, fontSize: 10, fill: C.muted }}
                  tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.border }} tickLine={false}
                />
                <Tooltip contentStyle={TS}
                  formatter={(v: any, name: string) =>
                    name === "θ" ? [`${v}°`, "θ"] : [`${v} cm/s`, "Vc"]
                  }
                />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Scatter data={annotPts} fill={C.accent} opacity={0.65} name="Annotations" />
                {regAnnotLine.length > 0 && (
                  <Scatter data={regAnnotLine} line={{ stroke: C.accent, strokeWidth: 1.5 }}
                    shape={() => null as any} legendType="none" name="rég. annot." />
                )}
                {nnPts.length > 0 && (
                  <Scatter data={nnPts} fill={C.green} opacity={0.85} name="Modèle NN" />
                )}
                {regNNLine.length > 0 && (
                  <Scatter data={regNNLine}
                    line={{ stroke: C.green, strokeWidth: 2, strokeDasharray: "6 3" }}
                    shape={() => null as any} legendType="none" name="rég. modèle" />
                )}
              </ScatterChart>
            </ResponsiveContainer>
          </>
        )}
      </Section>

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
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
                  <MCard label="MAE" value={`${progress.mae} cm/s`} color={C.accent} />
                  <MCard label="RMSE" value={`${progress.rmse} cm/s`} color={C.blue} />
                  <MCard label="Dataset" value={`${progress.n_train}+${progress.n_val}`} color={C.green} sub="train+val" />
                </div>

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

                {scatterData.length > 0 && (
                  <div style={{ marginBottom: 20 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: C.muted, marginBottom: 4 }}>Vc prédit vs réel (cm/s)</div>
                    <div style={{ fontSize: 10, color: C.muted, marginBottom: 8 }}>
                      Chaque point = une image de validation · droite pointillée = prédiction parfaite (y = x)
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                      <ScatterChart margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis
                          dataKey="x" type="number" name="Vc réel"
                          domain={[axMin, axMax]} tickCount={6}
                          label={{ value: "Vc réel (cm/s)", position: "insideBottom", offset: -14, fontSize: 10, fill: C.muted }}
                          tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.border }} tickLine={false}
                        />
                        <YAxis
                          dataKey="y" type="number" name="Vc prédit"
                          domain={[axMin, axMax]} tickCount={6}
                          label={{ value: "Vc prédit (cm/s)", angle: -90, position: "insideLeft", offset: 14, fill: C.muted, fontSize: 10 }}
                          tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.border }} tickLine={false}
                        />
                        <Tooltip contentStyle={TS} formatter={(v: any) => [`${v} cm/s`]} />
                        <Scatter data={diagPts} line={{ stroke: "#888", strokeDasharray: "5 3", strokeWidth: 1.5 }}
                          shape={() => null as any} legendType="none" name="y=x" />
                        <Scatter data={scatterData} fill={C.green} opacity={0.85} name="Mesures" />
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

      {/* MODÈLES DISPONIBLES — avec bouton Visualiser */}
      {models.length > 0 && (
        <Section icon="📦" title="Modèles entraînés">
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {models.map(m => (
              <div key={m.name}>
                <div style={{
                  display: "flex", justifyContent: "space-between", alignItems: "center",
                  padding: "10px 14px", background: C.bg, borderRadius: vizModel === m.name ? "10px 10px 0 0" : 10,
                  border: `1px solid ${vizModel === m.name ? C.accent : C.border}`,
                  transition: "border-color 0.2s",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span style={{ fontSize: 16 }}>{m.is_global ? "🌍" : "📂"}</span>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: 13 }}>{m.name}</div>
                      <div style={{ fontSize: 10, color: C.muted }}>{m.size_mb} MB — {new Date(m.modified_at).toLocaleDateString("fr-FR")}</div>
                    </div>
                  </div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <button
                      onClick={() => handleVisualize(m.name)}
                      style={{
                        fontSize: 11, padding: "4px 12px", borderRadius: 6, cursor: "pointer",
                        border: `1px solid ${vizModel === m.name ? C.accent : C.border}`,
                        background: vizModel === m.name ? "rgba(0,255,255,0.12)" : C.surface2,
                        color: vizModel === m.name ? C.accent : C.text,
                        fontWeight: 600, transition: "all 0.2s",
                      }}
                    >
                      {vizLoading && vizModel === m.name ? "⏳" : vizModel === m.name ? "✕ Fermer" : "📈 Visualiser"}
                    </button>
                    <a
                      href={`http://localhost:8000/api/models/${encodeURIComponent(m.filename)}/download`}
                      className="btn btn-secondary"
                      style={{ fontSize: 11, textDecoration: "none", padding: "4px 12px" }}
                      download={m.filename}
                    >
                      ⬇ Télécharger
                    </a>
                  </div>
                </div>

                {/* Panneau de visualisation dépliable */}
                {vizModel === m.name && (
                  <div style={{
                    padding: 16, background: "rgba(0,255,255,0.03)",
                    border: `1px solid ${C.accent}`, borderTop: "none",
                    borderRadius: "0 0 10px 10px",
                  }}>
                    {vizLoading ? (
                      <div style={{ textAlign: "center", color: C.muted, fontSize: 13, padding: 24 }}>⏳ Chargement des données…</div>
                    ) : vizPts.length < 2 ? (
                      <div style={{ textAlign: "center", color: C.muted, fontSize: 13, padding: 16 }}>
                        Pas assez de données Vc/θ pour ce modèle.
                      </div>
                    ) : (
                      <>
                        <div style={{ fontSize: 11, color: C.muted, marginBottom: 8 }}>
                          <span style={{ color: C.accent }}>● Données réelles</span>
                          {" : Vc = "}{regViz.a.toFixed(3)}·θ {regViz.b >= 0 ? "+" : ""}{regViz.b.toFixed(2)} cm/s
                          <span style={{ marginLeft: 8 }}>R² = {regViz.r2.toFixed(3)}</span>
                          <span style={{ marginLeft: 8, color: C.muted }}>({vizPts.length} points)</span>
                        </div>
                        <ResponsiveContainer width="100%" height={220}>
                          <ScatterChart margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                            <XAxis
                              dataKey="x" type="number" name="θ"
                              domain={[vizThMin, vizThMax]} tickCount={7}
                              label={{ value: "θ (°)", position: "insideBottom", offset: -14, fontSize: 10, fill: C.muted }}
                              tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.border }} tickLine={false}
                            />
                            <YAxis
                              dataKey="y" type="number" name="Vc"
                              domain={[vizVcMin, vizVcMax]} tickCount={6}
                              label={{ value: "Vc (cm/s)", angle: -90, position: "insideLeft", offset: 14, fontSize: 10, fill: C.muted }}
                              tick={{ fontSize: 9, fill: C.muted }} axisLine={{ stroke: C.border }} tickLine={false}
                            />
                            <Tooltip contentStyle={TS}
                              formatter={(v: any, name: string) =>
                                name === "θ" ? [`${v}°`, "θ"] : [`${v} cm/s`, "Vc"]
                              }
                            />
                            <Legend wrapperStyle={{ fontSize: 10 }} />
                            <Scatter data={vizPts} fill={C.accent} opacity={0.65} name="Annotations" />
                            {regVizLine.length > 0 && (
                              <Scatter data={regVizLine} line={{ stroke: C.accent, strokeWidth: 1.5 }}
                                shape={() => null as any} legendType="none" name="régression" />
                            )}
                          </ScatterChart>
                        </ResponsiveContainer>
                      </>
                    )}
                  </div>
                )}
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
