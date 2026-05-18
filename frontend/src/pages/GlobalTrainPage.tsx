import { useEffect, useRef, useState } from "react";
import { useLocation } from "wouter";
import {
  listSessions, listModels, trainGlobalModel, getGlobalTrainProgress, predictVc,
  type SessionMeta, type ModelMeta, type TrainProgress,
} from "../lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ComposedChart, Scatter, ResponsiveContainer, ReferenceLine,
} from "recharts";

const TS = {
  background: "var(--color-surface)",
  border: "1px solid var(--color-border)",
  borderRadius: 8, fontSize: 12, color: "var(--color-text)",
};

function linReg(pts: {x: number, y: number}[]) {
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

// Ticks uniformément espacés entre min et max
function niceTicks(min: number, max: number, n = 6): number[] {
  const step = (max - min) / (n - 1);
  return Array.from({ length: n }, (_, i) => parseFloat((min + step * i).toFixed(2)));
}

export default function GlobalTrainPage() {
  const [, navigate] = useLocation();
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [epochs, setEpochs] = useState(50);
  const [modelName, setModelName] = useState("global_vc_model");
  const [progress, setProgress] = useState<TrainProgress | null>(null);
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [selModel, setSelModel] = useState("");
  const [predFile, setPredFile] = useState<File | null>(null);
  const [predResult, setPredResult] = useState<number | null>(null);
  const [predError, setPredError] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [vcSource, setVcSource] = useState("");
  const [vcLoading, setVcLoading] = useState(false);
  const [angleVcData, setAngleVcData] = useState<{theta: number, vc: number}[]>([]);
  const pollRef = useRef<any>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadAngleVc = (source: string) => {
    setVcLoading(true);
    let url = "/api/train/global/angle-vc-data";
    if (source) {
      if (source.startsWith("folder:")) {
        url += `?folder=${encodeURIComponent(source.replace("folder:", ""))}`;
      } else {
        url += `?sessions=${encodeURIComponent(source)}`;
      }
    }
    fetch(url)
      .then(r => r.ok ? r.json() : [])
      .then(d => setAngleVcData(d))
      .catch(() => {})
      .finally(() => setVcLoading(false));
  };

  useEffect(() => {
    listSessions().then(s => setSessions(s)).catch(() => {});
    loadModels();
    getGlobalTrainProgress().then(p => {
      if (p.status === "running" || p.status === "done") setProgress(p);
    }).catch(() => {});
    loadAngleVc(vcSource);
  }, [vcSource]);

  const loadModels = async () => {
    try {
      const m = await listModels();
      setModels(m);
      const g = m.find(x => x.is_global);
      if (g) setSelModel(g.name);
      else if (m.length > 0) setSelModel(m[0].name);
    } catch { /* ignore */ }
  };

  const startPolling = () => {
    pollRef.current = setInterval(async () => {
      try {
        const p = await getGlobalTrainProgress();
        setProgress(p);
        if (p.status === "done" || p.status === "error") {
          clearInterval(pollRef.current);
          loadModels();
          loadAngleVc();
        }
      } catch { clearInterval(pollRef.current); }
    }, 800);
  };

  const handleTrain = async () => {
    try {
      setProgress({ epoch: 0, total_epochs: epochs, status: "starting" });
      await trainGlobalModel({
        sessions: selected.length > 0 ? selected.join(",") : undefined,
        epochs, model_name: modelName,
      });
      startPolling();
    } catch (e: any) {
      setProgress({ epoch: 0, total_epochs: epochs, status: "error", error: e.message });
    }
  };

  const handlePredict = async () => {
    if (!predFile || !selModel) return;
    setPredicting(true); setPredError(""); setPredResult(null);
    try {
      const r = await predictVc(selModel, predFile);
      setPredResult(r.vitesse_estimee);
    } catch (e: any) { setPredError(e.message); }
    finally { setPredicting(false); }
  };

  const pct = progress && progress.total_epochs > 0
    ? Math.round((progress.epoch / progress.total_epochs) * 100) : 0;

  const lossData = progress?.train_losses?.map((tl, i) => ({
    epoch: i + 1, train: tl, val: progress.val_losses?.[i] ?? 0,
  })) ?? [];

  // ── Scatter Prédit vs Réel ─────────────────────────────────────────────────
  // On trie par Vc réel pour que l'axe X soit ordonné
  const scatterRaw = progress?.preds?.map((p, i) => ({
    x: parseFloat((progress.true?.[i] ?? 0).toFixed(2)),
    y: parseFloat(p.toFixed(2)),
  })) ?? [];
  const scatterData = [...scatterRaw].sort((a, b) => a.x - b.x);

  const allVals = scatterData.flatMap(d => [d.x, d.y]);
  const rawMin = allVals.length > 0 ? Math.min(...allVals) : 0;
  const rawMax = allVals.length > 0 ? Math.max(...allVals) : 40;
  const marginSc = (rawMax - rawMin) * 0.12 + 1;
  const axMin = parseFloat((rawMin - marginSc).toFixed(1));
  const axMax = parseFloat((rawMax + marginSc).toFixed(1));
  const scTicks = niceTicks(axMin, axMax, 6);

  // ── Graphique Vc = f(θ) — nuage annoté + modèle NN ─────────────────────────
  const annotPts = [...angleVcData]
    .sort((a, b) => a.theta - b.theta)
    .map(d => ({ x: d.theta, y: d.vc }));

  const regAnnot = linReg(annotPts);

  const modelPredPts: {x: number, y: number}[] = [];
  if (progress?.preds && progress.preds.length > 0 && angleVcData.length > 0) {
    const nPreds = progress.preds.length;
    const slice = angleVcData.slice(-nPreds);
    slice.forEach((d, i) => {
      modelPredPts.push({ x: d.theta, y: parseFloat((progress.preds![i]).toFixed(2)) });
    });
    modelPredPts.sort((a, b) => a.x - b.x);
  }
  const regModel = linReg(modelPredPts);

  const allTheta = [...annotPts, ...modelPredPts].map(p => p.x);
  const allVc    = [...annotPts, ...modelPredPts].map(p => p.y);
  const tMin = allTheta.length > 0 ? parseFloat((Math.min(...allTheta) - 2).toFixed(1)) : 0;
  const tMax = allTheta.length > 0 ? parseFloat((Math.max(...allTheta) + 2).toFixed(1)) : 90;
  const vMin = allVc.length > 0   ? parseFloat((Math.min(...allVc) - 2).toFixed(1)) : 0;
  const vMax = allVc.length > 0   ? parseFloat((Math.max(...allVc) + 2).toFixed(1)) : 40;
  const tTicks = niceTicks(tMin, tMax, 7);
  const vTicks = niceTicks(vMin, vMax, 7);

  // Droite régression annoté : 2 points suffisent pour une droite
  const regAnnotLine = [
    { x: tMin, yAnnot: parseFloat((regAnnot.a * tMin + regAnnot.b).toFixed(2)) },
    { x: tMax, yAnnot: parseFloat((regAnnot.a * tMax + regAnnot.b).toFixed(2)) },
  ];
  // Droite modèle NN
  const regModelLine = modelPredPts.length >= 2 ? [
    { x: tMin, yModel: parseFloat((regModel.a * tMin + regModel.b).toFixed(2)) },
    { x: tMax, yModel: parseFloat((regModel.a * tMax + regModel.b).toFixed(2)) },
  ] : [];

  // Merge des datasets pour ComposedChart
  // Pour Vc=f(θ) on doit avoir un tableau commun indexé par x
  // On utilise deux Scatter séparés dans ComposedChart (ils partagent les mêmes axes numériques)
  const totalAnnotated = sessions.reduce((acc, s) =>
    acc + s.images.filter(i => i.status === "annotated").length, 0);

  return (
    <div className="animate-fade-in space-y-8">
      {/* Header */}
      <div>
        <button onClick={() => navigate("/")} className="text-xs text-dim hover:text-text mb-3 flex items-center gap-1 transition-colors">
          ← Retour aux sessions
        </button>
        <h1 className="text-3xl font-black tracking-tight">🌐 Modèle global</h1>
        <p className="text-sm text-dim mt-1">
          Entraînez un modèle sur plusieurs sessions en même temps pour une meilleure généralisation.
        </p>
      </div>

      {/* Stats */}
      <div className="card flex gap-8 py-5">
        <div><div className="text-2xl font-black">{sessions.length}</div><div className="text-[10px] uppercase tracking-widest text-dim mt-0.5">Sessions</div></div>
        <div><div className="text-2xl font-black text-green">{totalAnnotated}</div><div className="text-[10px] uppercase tracking-widest text-dim mt-0.5">Images annotées</div></div>
        <div><div className="text-2xl font-black text-accent">{models.length}</div><div className="text-[10px] uppercase tracking-widest text-dim mt-0.5">Modèles dispo</div></div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Config */}
        <div className="card space-y-5">
          <h3 className="text-base font-bold">⚙️ Configuration</h3>
          <div>
            <label className="block text-xs font-bold text-dim uppercase tracking-wider mb-2">Sessions (tout si vide)</label>
            <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
              {sessions.map(s => {
                const ann = s.images.filter(i => i.status === "annotated").length;
                return (
                  <label key={s.name} className="flex items-center gap-2 cursor-pointer hover:bg-surface-hover p-1.5 rounded-lg transition-colors">
                    <input type="checkbox" className="accent-accent"
                      checked={selected.includes(s.name)}
                      onChange={e => setSelected(prev => e.target.checked ? [...prev, s.name] : prev.filter(x => x !== s.name))}
                    />
                    <span className="text-sm flex-1">{s.name}</span>
                    <span className="text-[10px] text-dim">{ann} ann.</span>
                  </label>
                );
              })}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-dim mb-1.5">Époques</label>
              <input type="number" value={epochs} onChange={e => setEpochs(parseInt(e.target.value) || 10)} min={1} max={500} />
            </div>
            <div>
              <label className="block text-xs text-dim mb-1.5">Nom du modèle</label>
              <input type="text" value={modelName} onChange={e => setModelName(e.target.value)} />
            </div>
          </div>
          <button
            className="btn btn-primary w-full justify-center"
            onClick={handleTrain}
            disabled={progress?.status === "running" || progress?.status === "starting"}
          >
            {progress?.status === "running" || progress?.status === "starting"
              ? `⏳ Epoch ${progress?.epoch}/${progress?.total_epochs}...`
              : "🧠 Lancer l'entraînement global"}
          </button>
          {progress && progress.status !== "idle" && (
            <div className="space-y-2 animate-fade-in">
              <div className="flex justify-between text-xs">
                <span className={progress.status === "error" ? "text-red" : progress.status === "done" ? "text-green font-bold" : "text-accent"}>
                  {progress.status === "error" ? `❌ ${progress.error}` :
                   progress.status === "done" ? `✅ MAE: ${progress.mae} cm/s · RMSE: ${progress.rmse} cm/s` :
                   `Epoch ${progress.epoch} / ${progress.total_epochs}`}
                </span>
                {progress.device && <span className="text-dim">{progress.device}</span>}
              </div>
              {progress.status !== "error" && (
                <div className="progress-bar"><div className="progress-fill" style={{ width: `${pct}%` }} /></div>
              )}
            </div>
          )}
        </div>

        {/* Prédiction */}
        <div className="card space-y-5">
          <h3 className="text-base font-bold">🌊 Prédire Vc</h3>
          {models.length === 0 ? (
            <div className="text-sm text-dim">Aucun modèle disponible.</div>
          ) : (
            <div className="space-y-3">
              <select value={selModel} onChange={e => setSelModel(e.target.value)}>
                {models.map(m => <option key={m.name} value={m.name}>{m.is_global ? "🌐 " : ""}{m.name} ({m.size_mb} MB)</option>)}
              </select>
              <div
                className={`drop-zone ${dragOver ? "drag-over" : ""} flex flex-col items-center justify-center`}
                style={{ minHeight: 80 }}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) setPredFile(f); }}
                onClick={() => fileRef.current?.click()}
              >
                <input ref={fileRef} type="file" accept="image/*" hidden onChange={e => e.target.files?.[0] && setPredFile(e.target.files[0])} />
                {predFile ? <span className="text-sm font-bold text-accent">✓ {predFile.name}</span> :
                  <><div className="text-2xl mb-1">🖼</div><p className="text-xs text-dim">Glissez ou cliquez</p></>}
              </div>
              <button className="btn btn-primary w-full justify-center" disabled={!predFile || predicting} onClick={handlePredict}>
                {predicting ? "⏳ Analyse..." : "🔍 Analyser"}
              </button>
              {predResult !== null && (
                <div className="p-4 rounded-xl border border-accent/30 bg-accent/5 text-center animate-fade-in">
                  <div className="text-3xl font-black text-accent">{predResult} cm/s</div>
                  <div className="text-xs text-dim mt-1">🌊 Vitesse estimée</div>
                </div>
              )}
              {predError && <div className="text-sm text-red">❌ {predError}</div>}
            </div>
          )}
        </div>
      </div>

      {/* Charts post-entraînement */}
      {progress?.status === "done" && lossData.length > 0 && (
        <div className="grid grid-cols-2 gap-6 animate-fade-in">

          {/* Courbe de loss */}
          <div className="card">
            <h4 className="text-sm font-bold mb-4">📉 Courbe de loss</h4>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                <XAxis dataKey="epoch"
                  label={{ value: "Époque", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "var(--color-text-dim)" }}
                  tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false}
                />
                <YAxis tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={TS} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="train" stroke="var(--color-accent)" dot={false} name="Train Loss" strokeWidth={2} />
                <Line type="monotone" dataKey="val" stroke="var(--color-green)" dot={false} name="Val Loss" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Vc prédit vs réel — ComposedChart orthonormé */}
          {scatterData.length > 0 && (
            <div className="card">
              <h4 className="text-sm font-bold mb-1">🎯 Vc prédit vs réel (cm/s)</h4>
              <p className="text-[10px] text-dim mb-3">
                Chaque point = une image · droite pointillée = prédiction parfaite (y = x)
              </p>
              <ResponsiveContainer width="100%" height={260}>
                <ComposedChart margin={{ top: 10, right: 20, bottom: 35, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    type="number"
                    dataKey="x"
                    domain={[axMin, axMax]}
                    ticks={scTicks}
                    tickFormatter={v => v.toFixed(1)}
                    label={{ value: "Vc réel (cm/s)", position: "insideBottom", offset: -20, fontSize: 11, fill: "var(--color-text-dim)" }}
                    tick={{ fontSize: 10, fill: "var(--color-text-dim)" }}
                    axisLine={{ stroke: "var(--color-border)" }} tickLine={false}
                  />
                  <YAxis
                    type="number"
                    dataKey="y"
                    domain={[axMin, axMax]}
                    ticks={scTicks}
                    tickFormatter={v => v.toFixed(1)}
                    label={{ value: "Vc prédit (cm/s)", angle: -90, position: "insideLeft", offset: 15, fontSize: 11, fill: "var(--color-text-dim)" }}
                    tick={{ fontSize: 10, fill: "var(--color-text-dim)" }}
                    axisLine={{ stroke: "var(--color-border)" }} tickLine={false}
                  />
                  <Tooltip contentStyle={TS} formatter={(v: any) => [`${v} cm/s`]} />
                  {/* Diagonale y=x via ReferenceLine */}
                  <ReferenceLine
                    segment={[{ x: axMin, y: axMin }, { x: axMax, y: axMax }]}
                    stroke="#888" strokeDasharray="6 3" strokeWidth={1.5}
                    ifOverflow="extendDomain"
                  />
                  {/* Nuage de points mesures */}
                  <Scatter
                    data={scatterData}
                    fill="var(--color-accent)"
                    opacity={0.85}
                    name="Mesures"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Graphique Vc = f(θ) — toujours visible dès qu'il y a des données */}
      <div className="card animate-fade-in">
        <div className="flex justify-between items-center mb-4">
          <h4 className="text-sm font-bold">📐 Vc = f(θ) — angle PCA du câble</h4>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-dim uppercase font-bold">Source :</span>
            <select
              className="text-xs p-1 h-8 min-w-[180px]"
              value={vcSource}
              onChange={e => setVcSource(e.target.value)}
            >
              <option value="">🌍 Toutes les sessions</option>
              {Array.from(new Set(sessions.map(s => s.folder).filter(Boolean))).map(f => (
                <option key={`folder:${f}`} value={`folder:${f}`}>📁 Dossier: {f}</option>
              ))}
              <optgroup label="── Par session ──">
                {sessions.map(s => (
                  <option key={s.name} value={s.name}>📂 {s.name}</option>
                ))}
              </optgroup>
            </select>
            {vcLoading && <span className="text-[10px] text-accent animate-pulse">⏳</span>}
          </div>
        </div>

        {annotPts.length < 3 ? (
          <div className="py-12 text-center text-dim text-xs border border-dashed border-border rounded-xl">
            Pas assez de données pour cette source.
          </div>
        ) : (
          <>
            <p className="text-[10px] text-dim mb-1">
              <span className="text-accent font-mono">🔵 Annotations : Vc = {regAnnot.a.toFixed(3)}·θ {regAnnot.b >= 0 ? "+" : ""}{regAnnot.b.toFixed(2)} cm/s</span>
              <span className="ml-2 text-dim">(R² = {regAnnot.r2.toFixed(3)})</span>
            </p>
            {regModelLine.length > 0 && (
              <p className="text-[10px] text-dim mb-3">
                <span className="text-green font-mono">🟢 Modèle NN : Vc = {regModel.a.toFixed(3)}·θ {regModel.b >= 0 ? "+" : ""}{regModel.b.toFixed(2)} cm/s</span>
                <span className="ml-2 text-dim">(R² = {regModel.r2.toFixed(3)})</span>
              </p>
            )}
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart margin={{ top: 10, right: 20, bottom: 40, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis
                  type="number"
                  dataKey="x"
                  domain={[tMin, tMax]}
                  ticks={tTicks}
                  tickFormatter={v => v.toFixed(1)}
                  label={{ value: "θ (°)", position: "insideBottom", offset: -25, fontSize: 11, fill: "var(--color-text-dim)" }}
                  tick={{ fontSize: 10, fill: "var(--color-text-dim)" }}
                  axisLine={{ stroke: "var(--color-border)" }} tickLine={false}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  domain={[vMin, vMax]}
                  ticks={vTicks}
                  tickFormatter={v => v.toFixed(1)}
                  label={{ value: "Vc (cm/s)", angle: -90, position: "insideLeft", offset: 15, fontSize: 11, fill: "var(--color-text-dim)" }}
                  tick={{ fontSize: 10, fill: "var(--color-text-dim)" }}
                  axisLine={{ stroke: "var(--color-border)" }} tickLine={false}
                />
                <Tooltip
                  contentStyle={TS}
                  formatter={(v: any, name: string) =>
                    name === "θ" ? [`${v}°`, "θ"] : [`${v} cm/s`, name]
                  }
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />

                {/* Nuage — annotations */}
                <Scatter data={annotPts} fill="var(--color-accent)" opacity={0.65} name="Annotations" />

                {/* Droite régression — annotations */}
                <Line
                  data={regAnnotLine}
                  dataKey="yAnnot"
                  dot={false}
                  stroke="var(--color-accent)"
                  strokeWidth={2}
                  legendType="none"
                  name="Régression annot."
                  isAnimationActive={false}
                />

                {/* Nuage — prédictions modèle NN */}
                {modelPredPts.length > 0 && (
                  <Scatter data={modelPredPts} fill="var(--color-green)" opacity={0.8} name="Modèle NN" />
                )}

                {/* Droite régression — modèle NN */}
                {regModelLine.length > 0 && (
                  <Line
                    data={regModelLine}
                    dataKey="yModel"
                    dot={false}
                    stroke="var(--color-green)"
                    strokeWidth={2}
                    strokeDasharray="6 3"
                    legendType="none"
                    name="Régression NN"
                    isAnimationActive={false}
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </>
        )}
      </div>
    </div>
  );
}
