import { useEffect, useRef, useState, useCallback } from "react";
import {
  trainSessionModel, getTrainProgress, listModels, predictVc,
  type TrainProgress, type ModelMeta,
} from "../lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ScatterChart, Scatter, ResponsiveContainer,
} from "recharts";

const TOOLTIP_STYLE = {
  background: "var(--color-surface)",
  border: "1px solid var(--color-border)",
  borderRadius: 8,
  fontSize: 12,
  color: "var(--color-text)",
};

export default function AIPanel({ sessionName }: { sessionName: string }) {
  const [epochs, setEpochs] = useState(50);
  const [progress, setProgress] = useState<TrainProgress | null>(null);
  const [models, setModels] = useState<ModelMeta[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [predFile, setPredFile] = useState<File | null>(null);
  const [predResult, setPredResult] = useState<number | null>(null);
  const [predError, setPredError] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const pollRef = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadModels = useCallback(async () => {
    try {
      const m = await listModels();
      setModels(m);
      if (m.length > 0 && !selectedModel) setSelectedModel(m[0].name);
    } catch { /* ignore */ }
  }, [selectedModel]);

  useEffect(() => {
    loadModels();
    // Restore progress if already training
    getTrainProgress(sessionName).then(p => {
      if (p.status === "running" || p.status === "done") setProgress(p);
    }).catch(() => {});
  }, [sessionName, loadModels]);

  const startPolling = () => {
    pollRef.current = setInterval(async () => {
      try {
        const p = await getTrainProgress(sessionName);
        setProgress(p);
        if (p.status === "done" || p.status === "error") {
          clearInterval(pollRef.current);
          loadModels();
        }
      } catch { clearInterval(pollRef.current); }
    }, 800);
  };

  const handleTrain = async () => {
    try {
      setProgress({ epoch: 0, total_epochs: epochs, status: "starting" });
      await trainSessionModel(sessionName, epochs);
      startPolling();
    } catch (e: any) {
      setProgress({ epoch: 0, total_epochs: epochs, status: "error", error: e.message });
    }
  };

  const handlePredict = async () => {
    if (!predFile || !selectedModel) return;
    setPredicting(true);
    setPredError("");
    setPredResult(null);
    try {
      const r = await predictVc(selectedModel, predFile);
      setPredResult(r.vitesse_estimee);
    } catch (e: any) {
      setPredError(e.message);
    } finally {
      setPredicting(false);
    }
  };

  const pct = progress && progress.total_epochs > 0
    ? Math.round((progress.epoch / progress.total_epochs) * 100) : 0;

  const lossData = progress?.train_losses?.map((tl, i) => ({
    epoch: i + 1, train: tl, val: progress.val_losses?.[i] ?? 0,
  })) ?? [];

  const scatterData = progress?.preds?.map((p, i) => ({
    pred: parseFloat(p.toFixed(2)), true: parseFloat((progress.true?.[i] ?? 0).toFixed(2)),
  })) ?? [];

  return (
    <div className="space-y-8 animate-fade-in">

      {/* ── Entraînement ── */}
      <div className="card space-y-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold">🧠 Entraîner un modèle Vc</h3>
            <p className="text-xs text-dim mt-0.5">ResNet18 — régression image → vitesse courant (cm/s)</p>
          </div>
          <div className="flex items-center gap-3">
            <div>
              <label className="text-xs text-dim">Epochs</label>
              <input
                type="number" value={epochs}
                onChange={e => setEpochs(parseInt(e.target.value) || 10)}
                min={1} max={500}
                style={{ width: 72, marginLeft: 8 }}
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={handleTrain}
              disabled={progress?.status === "running" || progress?.status === "starting"}
            >
              {progress?.status === "running" || progress?.status === "starting"
                ? `⏳ Epoch ${progress.epoch}/${progress.total_epochs}...`
                : "🧠 Lancer l'entraînement"}
            </button>
          </div>
        </div>

        {/* Progress bar */}
        {progress && progress.status !== "idle" && (
          <div className="animate-fade-in space-y-2">
            <div className="flex justify-between text-xs">
              <span className={progress.status === "error" ? "text-red" : progress.status === "done" ? "text-green font-bold" : "text-accent"}>
                {progress.status === "error" ? `❌ ${progress.error}` :
                 progress.status === "done" ? `✅ Terminé — MAE: ${progress.mae} cm/s · RMSE: ${progress.rmse} cm/s` :
                 `Epoch ${progress.epoch} / ${progress.total_epochs}`}
              </span>
              {progress.status === "running" && (
                <span className="text-dim">{progress.device} · {progress.n_train} train / {progress.n_val} val</span>
              )}
            </div>
            {progress.status !== "error" && (
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${pct}%` }} />
              </div>
            )}
          </div>
        )}

        {/* Charts */}
        {progress?.status === "done" && lossData.length > 0 && (
          <div className="grid grid-cols-2 gap-4 mt-2 animate-fade-in">
            <div>
              <h4 className="text-xs font-bold text-dim mb-2">Courbes de loss</h4>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={lossData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                  <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Line type="monotone" dataKey="train" stroke="var(--color-accent)" dot={false} name="Train" strokeWidth={2} />
                  <Line type="monotone" dataKey="val" stroke="var(--color-green)" dot={false} name="Val" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {scatterData.length > 0 && (
              <div>
                <h4 className="text-xs font-bold text-dim mb-2">Vc prédit vs réel (cm/s)</h4>
                <ResponsiveContainer width="100%" height={180}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="true" name="Réel" tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false} />
                    <YAxis dataKey="pred" name="Prédit" tick={{ fontSize: 10, fill: "var(--color-text-dim)" }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={TOOLTIP_STYLE} />
                    <Scatter data={scatterData} fill="var(--color-accent)" opacity={0.75} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {/* No data hint */}
        {!progress && (
          <div className="text-xs text-dim p-3 bg-bg rounded-lg border border-border">
            ℹ️ Renseignez <strong>current_speed_cm_s</strong> sur au moins 2 images annotées pour activer l'entraînement.
          </div>
        )}
      </div>

      {/* ── Prédiction ── */}
      <div className="card space-y-5">
        <h3 className="text-base font-bold">🌊 Prédire Vc sur une image</h3>

        {models.length === 0 ? (
          <div className="text-sm text-dim p-4 bg-bg rounded-lg border border-border">
            Aucun modèle disponible. Annotez des images avec <strong>current_speed_cm_s</strong> puis cliquez Entraîner.
          </div>
        ) : (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-dim mb-1.5">Modèle</label>
              <select value={selectedModel} onChange={e => setSelectedModel(e.target.value)}>
                {models.map(m => (
                  <option key={m.name} value={m.name}>
                    {m.is_global ? "🌐 " : ""}{m.name} ({m.size_mb} MB)
                  </option>
                ))}
              </select>
            </div>

            <div
              className={`drop-zone ${dragOver ? "drag-over" : ""} flex flex-col items-center justify-center`}
              style={{ minHeight: 100 }}
              onDragOver={e => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) setPredFile(f); }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input ref={fileInputRef} type="file" accept="image/*" hidden onChange={e => e.target.files?.[0] && setPredFile(e.target.files[0])} />
              {predFile ? (
                <span className="text-sm font-bold text-accent">✓ {predFile.name}</span>
              ) : (
                <><div className="text-3xl mb-2">🖼</div><p className="text-xs text-dim">Glissez une image ou cliquez</p></>
              )}
            </div>

            <button
              className="btn btn-primary w-full justify-center"
              disabled={!predFile || predicting}
              onClick={handlePredict}
            >
              {predicting ? "⏳ Analyse..." : "🔍 Analyser"}
            </button>

            {predResult !== null && (
              <div className="animate-fade-in p-5 rounded-xl border border-accent/30 bg-accent/5 text-center">
                <div className="text-3xl font-black text-accent tabular-nums">{predResult} cm/s</div>
                <div className="text-xs text-dim mt-1">🌊 Vitesse estimée du courant</div>
                <div className="text-[10px] text-dim mt-0.5">{selectedModel}</div>
              </div>
            )}
            {predError && (
              <div className="text-sm text-red p-3 bg-red/5 rounded-lg border border-red/20">❌ {predError}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
