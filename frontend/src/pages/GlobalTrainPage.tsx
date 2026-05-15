import { useEffect, useRef, useState } from "react";
import { useLocation } from "wouter";
import {
  listSessions, listModels, trainGlobalModel, getGlobalTrainProgress, predictVc,
  type SessionMeta, type ModelMeta, type TrainProgress,
} from "../lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ScatterChart, Scatter, ResponsiveContainer, ReferenceLine,
} from "recharts";

const TS = {
  background: "var(--color-surface)",
  border: "1px solid var(--color-border)",
  borderRadius: 8, fontSize: 12, color: "var(--color-text)",
};

// Régression linéaire moindres carrés : y = a*x + b
function linReg(pts: {x: number, y: number}[]) {
  const n = pts.length;
  if (n < 2) return { a: 0, b: 0 };
  const sx = pts.reduce((s, p) => s + p.x, 0);
  const sy = pts.reduce((s, p) => s + p.y, 0);
  const sxx = pts.reduce((s, p) => s + p.x * p.x, 0);
  const sxy = pts.reduce((s, p) => s + p.x * p.y, 0);
  const a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  const b = (sy - a * sx) / n;
  return { a, b };
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
  const [angleVcData, setAngleVcData] = useState<{theta: number, vc: number}[]>([]);
  const pollRef = useRef<any>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    listSessions().then(s => setSessions(s)).catch(() => {});
    loadModels();
    getGlobalTrainProgress().then(p => {
      if (p.status === "running" || p.status === "done") setProgress(p);
    }).catch(() => {});
    // Charger données angle-vc
    fetch("/api/train/global/angle-vc-data")
      .then(r => r.ok ? r.json() : [])
      .then(d => setAngleVcData(d))
      .catch(() => {});
  }, []);

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
          // Recharger données angle-vc après entraînement
          fetch("/api/train/global/angle-vc-data")
            .then(r => r.ok ? r.json() : [])
            .then(d => setAngleVcData(d))
            .catch(() => {});
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

  // Scatter Prédit vs Réel : axe X = réel, axe Y = prédit
  const scatterData = progress?.preds?.map((p, i) => ({
    x: parseFloat((progress.true?.[i] ?? 0).toFixed(2)),
    y: parseFloat(p.toFixed(2)),
  })) ?? [];

  // Plage commune pour axe orthonormé
  const allVals = scatterData.flatMap(d => [d.x, d.y]);
  const axMin = allVals.length > 0 ? Math.floor(Math.min(...allVals)) : 0;
  const axMax = allVals.length > 0 ? Math.ceil(Math.max(...allVals)) : 40;
  // Droite diagonale y=x
  const diagData = [{ x: axMin, y: axMin }, { x: axMax, y: axMax }];

  // Données Vc = f(θ)
  const thetaPts = angleVcData.map(d => ({ x: d.theta, y: d.vc }));
  const reg = linReg(thetaPts);
  const thetaMin = thetaPts.length > 0 ? Math.floor(Math.min(...thetaPts.map(p => p.x))) : 0;
  const thetaMax = thetaPts.length > 0 ? Math.ceil(Math.max(...thetaPts.map(p => p.x))) : 90;
  const regLineData = [
    { x: thetaMin, y: parseFloat((reg.a * thetaMin + reg.b).toFixed(2)) },
    { x: thetaMax, y: parseFloat((reg.a * thetaMax + reg.b).toFixed(2)) },
  ];

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
              <label className="block text-xs text-dim mb-1.5">Epochs</label>
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

      {/* Charts */}
      {progress?.status === "done" && lossData.length > 0 && (
        <div className="grid grid-cols-2 gap-6 animate-fade-in">

          {/* Courbe de loss */}
          <div className="card">
            <h4 className="text-sm font-bold mb-4">📉 Courbe de loss</h4>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                <XAxis dataKey="epoch" label={{ value: "Époque", position: "insideBottomRight", offset: -5, fontSize: 10, fill: "var(--color-text-dim)" }} tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={TS} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="train" stroke="var(--color-accent)" dot={false} name="Train Loss" strokeWidth={2} />
                <Line type="monotone" dataKey="val" stroke="var(--color-green)" dot={false} name="Val Loss" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Scatter Prédit vs Réel — orthonormé + diagonale y=x */}
          {scatterData.length > 0 && (
            <div className="card">
              <h4 className="text-sm font-bold mb-1">🎯 Vc prédit vs réel (cm/s)</h4>
              <p className="text-[10px] text-dim mb-3">La droite pointillée = prédiction parfaite (y = x)</p>
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis
                    dataKey="x" type="number" name="Réel"
                    domain={[axMin, axMax]}
                    label={{ value: "Vc réel (cm/s)", position: "insideBottom", offset: -10, fontSize: 10, fill: "var(--color-text-dim)" }}
                    tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false}
                  />
                  <YAxis
                    dataKey="y" type="number" name="Prédit"
                    domain={[axMin, axMax]}
                    label={{ value: "Vc prédit (cm/s)", angle: -90, position: "insideLeft", offset: 10, fontSize: 10, fill: "var(--color-text-dim)" }}
                    tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false}
                  />
                  <Tooltip contentStyle={TS} formatter={(v: any, n: string) => [`${v} cm/s`, n === "x" ? "Réel" : "Prédit"]} />
                  {/* Diagonale y=x */}
                  <Scatter data={diagData} line={{ stroke: "#888", strokeDasharray: "6 3", strokeWidth: 1.5 }} shape={() => null as any} legendType="none" />
                  {/* Points mesurés */}
                  <Scatter data={scatterData} fill="var(--color-accent)" opacity={0.8} name="Mesures" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Graphique Vc = f(θ) */}
      {thetaPts.length >= 3 && (
        <div className="card animate-fade-in">
          <h4 className="text-sm font-bold mb-1">📐 Vc en fonction de θ (angle du câble)</h4>
          <p className="text-[10px] text-dim mb-3">
            Nuage de points de toutes les annotations · Droite de régression linéaire :
            <span className="font-mono text-accent ml-1">
              Vc = {reg.a >= 0 ? "+" : ""}{reg.a.toFixed(3)}·θ + {reg.b.toFixed(2)} cm/s
            </span>
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ top: 5, right: 20, bottom: 25, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
              <XAxis
                dataKey="x" type="number" name="θ"
                domain={[thetaMin, thetaMax]}
                label={{ value: "θ (°)", position: "insideBottom", offset: -10, fontSize: 11, fill: "var(--color-text-dim)" }}
                tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false}
              />
              <YAxis
                dataKey="y" type="number" name="Vc"
                label={{ value: "Vc (cm/s)", angle: -90, position: "insideLeft", offset: 10, fontSize: 11, fill: "var(--color-text-dim)" }}
                tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false}
              />
              <Tooltip contentStyle={TS} formatter={(v: any, n: string) => [n === "x" ? `${v}°` : `${v} cm/s`, n === "x" ? "θ" : "Vc"]} />
              {/* Droite de régression */}
              <Scatter data={regLineData} line={{ stroke: "var(--color-green)", strokeWidth: 2 }} shape={() => null as any} legendType="none" />
              {/* Points annotations */}
              <Scatter data={thetaPts} fill="var(--color-accent)" opacity={0.7} name="Annotations" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
