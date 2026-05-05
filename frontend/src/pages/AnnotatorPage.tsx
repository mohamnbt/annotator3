import { useState, useEffect, useRef, useCallback } from "react";
import { useLocation } from "wouter";
import {
  getSession, getAnnotation, saveAnnotation, ignoreImage, deleteImage,
  getLastConditions, imageUrl,
  type SessionMeta, type SessionImage, type Conditions, type AnnotationData,
} from "../lib/api";
import ConditionsPanel from "../components/ConditionsPanel";
import ShortcutsPanel from "../components/ShortcutsPanel";

interface Point { x: number; y: number }

function stemOf(filename: string) {
  return filename.replace(/\.[^.]+$/, "");
}

export default function AnnotatorPage({ sessionName }: { sessionName: string }) {
  const [session, setSession] = useState<SessionMeta | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [points, setPoints] = useState<Point[]>([]);
  const [mode, setMode] = useState<"centerline" | "contour">("centerline");
  const [activeSide, setActiveSide] = useState<"left" | "right">("left");
  const [leftPoints, setLeftPoints] = useState<Point[]>([]);
  const [rightPoints, setRightPoints] = useState<Point[]>([]);
  const [conditions, setConditions] = useState<Conditions>({});
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const [toast, setToast] = useState<string | null>(null);
  const [copiedFeedback, setCopiedFeedback] = useState(false);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [isRevision, setIsRevision] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const isSpacePressedRef = useRef(false);
  const [, navigate] = useLocation();

  // Load session
  useEffect(() => {
    getSession(sessionName).then((s) => {
      setSession(s);
      const firstUnannotated = s.images.findIndex((i) => i.status === "to_annotate");
      setCurrentIdx(firstUnannotated >= 0 ? firstUnannotated : 0);
    }).catch(() => navigate("/"));
  }, [sessionName, navigate]);

  // Load annotator name from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("cosmer_annotator_name");
    if (saved) setConditions((c) => ({ ...c, annotator_name: saved }));
  }, []);

  const currentImage: SessionImage | undefined = session?.images[currentIdx];
  const filename = currentImage?.filename || "";
  const stem = stemOf(filename);

  // Load image + existing annotation
  useEffect(() => {
    if (!session || !filename) return;
    setPoints([]);
    setLeftPoints([]);
    setRightPoints([]);
    setIsRevision(false);

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgRef.current = img;
      setImgSize({ w: img.naturalWidth, h: img.naturalHeight });
      setZoom(1);
      setPan({ x: 0, y: 0 });

      // Check for existing annotation
      getAnnotation(sessionName, stem).then((ann) => {
        if (ann && ann.points) {
          setIsRevision(true);
          setPoints(ann.points || []);
          if (ann.annotation_mode === "contour") {
            setMode("contour");
            setLeftPoints(ann.left_points || []);
            setRightPoints(ann.right_points || []);
          } else {
            setMode("centerline");
          }
          if (ann.conditions) {
            setConditions((prev) => ({ ...prev, ...ann.conditions }));
          }
        }
      }).catch(() => {});
    };
    img.src = imageUrl(sessionName, filename);
  }, [session, currentIdx, filename, sessionName, stem]);

  // Draw canvas
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !imgSize.w) return;

    const container = containerRef.current;
    if (!container) return;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    // Calculate fitted image dimensions
    const scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
    const iw = img.naturalWidth * scale;
    const ih = img.naturalHeight * scale;
    const ix = (canvas.width / zoom - iw) / 2;
    const iy = (canvas.height / zoom - ih) / 2;

    ctx.drawImage(img, ix, iy, iw, ih);

    const drawPolyline = (pts: Point[], color: string, firstColor: string) => {
      if (pts.length === 0) return;
      // Lines
      if (pts.length > 1) {
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 / zoom;
        ctx.moveTo(ix + (pts[0].x / img.naturalWidth) * iw, iy + (pts[0].y / img.naturalHeight) * ih);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(ix + (pts[i].x / img.naturalWidth) * iw, iy + (pts[i].y / img.naturalHeight) * ih);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      // Points
      pts.forEach((p, i) => {
        const px = ix + (p.x / img.naturalWidth) * iw;
        const py = iy + (p.y / img.naturalHeight) * ih;
        ctx.beginPath();
        ctx.arc(px, py, 4 / zoom, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? firstColor : color;
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1 / zoom;
        ctx.stroke();
      });
    };

    if (mode === "centerline") {
      drawPolyline(points, "#00FFFF", "#00FF88");
    } else {
      drawPolyline(leftPoints, "#00FFFF", "#00FF88");
      drawPolyline(rightPoints, "#FFA500", "#FFD700");
    }

    ctx.restore();
  }, [points, leftPoints, rightPoints, mode, zoom, pan, imgSize]);

  useEffect(() => { draw(); }, [draw]);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(container);
    return () => ro.disconnect();
  }, [draw]);

  // Convert screen coords to image coords
  const screenToImage = (ex: number, ey: number): Point | null => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return null;
    const rect = canvas.getBoundingClientRect();
    const mx = (ex - rect.left - pan.x) / zoom;
    const my = (ey - rect.top - pan.y) / zoom;
    const scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
    const iw = img.naturalWidth * scale;
    const ih = img.naturalHeight * scale;
    const ix = (canvas.width / zoom - iw) / 2;
    const iy = (canvas.height / zoom - ih) / 2;
    const imgX = ((mx - ix) / iw) * img.naturalWidth;
    const imgY = ((my - iy) / ih) * img.naturalHeight;
    if (imgX < 0 || imgY < 0 || imgX > img.naturalWidth || imgY > img.naturalHeight) return null;
    return { x: Math.round(imgX), y: Math.round(imgY) };
  };

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const pt = screenToImage(e.clientX, e.clientY);
    if (!pt) return;
    if (mode === "centerline") {
      setPoints((p) => [...p, pt]);
    } else {
      if (activeSide === "left") setLeftPoints((p) => [...p, pt]);
      else setRightPoints((p) => [...p, pt]);
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    if (mode === "centerline") {
      setPoints((p) => p.slice(0, -1));
    } else {
      if (activeSide === "left") setLeftPoints((p) => p.slice(0, -1));
      else setRightPoints((p) => p.slice(0, -1));
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.2, Math.min(10, zoom * delta));
    setPan({ x: mx - (mx - pan.x) * (newZoom / zoom), y: my - (my - pan.y) * (newZoom / zoom) });
    setZoom(newZoom);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && isSpacePressedRef.current)) {
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    }
  };

  const handleMouseUp = () => { setIsPanning(false); };

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 1500);
  };

  const goTo = (idx: number) => {
    if (!session || idx < 0 || idx >= session.images.length) return;
    setCurrentIdx(idx);
  };

  const goNextUnannotated = () => {
    if (!session) return;
    for (let i = currentIdx + 1; i < session.images.length; i++) {
      if (session.images[i].status === "to_annotate") { setCurrentIdx(i); return; }
    }
    for (let i = 0; i < currentIdx; i++) {
      if (session.images[i].status === "to_annotate") { setCurrentIdx(i); return; }
    }
    showToast("Toutes les images sont traitées !");
  };

  const handleSave = async () => {
    const activePoints = mode === "centerline" ? points : [...leftPoints, ...rightPoints];
    if (activePoints.length < 2) { showToast("Minimum 2 points requis"); return; }
    if (!session || !currentImage) return;

    // Save annotator name
    if (conditions.annotator_name) {
      localStorage.setItem("cosmer_annotator_name", conditions.annotator_name);
    }

    const data: AnnotationData = {
      points: mode === "centerline" ? points : [],
      left_points: mode === "contour" ? leftPoints : undefined,
      right_points: mode === "contour" ? rightPoints : undefined,
      annotation_mode: mode,
      conditions,
      image_width: imgSize.w,
      image_height: imgSize.h,
    };

    try {
      await saveAnnotation(sessionName, stem, data);
      showToast("Annotation sauvegardée ✓");
      // Reload session to update statuses
      const updated = await getSession(sessionName);
      setSession(updated);
      setTimeout(goNextUnannotated, 300);
    } catch (e: any) {
      showToast("Erreur: " + e.message);
    }
  };

  const handleIgnore = async () => {
    if (!currentImage) return;
    try {
      await ignoreImage(sessionName, filename);
      const updated = await getSession(sessionName);
      setSession(updated);
      showToast("Image ignorée");
      setTimeout(goNextUnannotated, 300);
    } catch (e: any) {
      showToast("Erreur: " + e.message);
    }
  };

  const handleDeleteImg = async () => {
    if (!currentImage || !confirm(`Supprimer ${filename} ?`)) return;
    try {
      await deleteImage(sessionName, filename);
      const updated = await getSession(sessionName);
      setSession(updated);
      if (currentIdx >= updated.images.length) setCurrentIdx(Math.max(0, updated.images.length - 1));
      showToast("Image supprimée");
    } catch (e: any) {
      showToast("Erreur: " + e.message);
    }
  };

  const handleCopyConditions = async () => {
    try {
      const last = await getLastConditions(sessionName);
      if (last) {
        setConditions((prev) => ({ ...prev, ...last }));
        setCopiedFeedback(true);
        setTimeout(() => setCopiedFeedback(false), 2000);
      } else {
        showToast("Aucune annotation précédente trouvée");
      }
    } catch {
      showToast("Erreur lors de la copie");
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if (e.key === "?") { setShowShortcuts((s) => !s); return; }
      if (e.key === "Escape") {
        if (showShortcuts) { setShowShortcuts(false); return; }
        setPoints([]); setLeftPoints([]); setRightPoints([]); return;
      }
      if (e.key === "z" || e.key === "Z") {
        if (mode === "centerline") setPoints((p) => p.slice(0, -1));
        else if (activeSide === "left") setLeftPoints((p) => p.slice(0, -1));
        else setRightPoints((p) => p.slice(0, -1));
        return;
      }
      if (e.key === "Enter") { handleSave(); return; }
      if (e.key === "ArrowRight") { goTo(currentIdx + 1); return; }
      if (e.key === "ArrowLeft") { goTo(currentIdx - 1); return; }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  // Space key for panning
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && (e.target as HTMLElement)?.tagName !== "INPUT") {
        e.preventDefault();
        document.body.style.cursor = "grab";
        isSpacePressedRef.current = true;
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") { 
        document.body.style.cursor = ""; 
        isSpacePressedRef.current = false;
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, []);

  if (!session || !currentImage) {
    return <div style={{ color: "var(--color-text-dim)", textAlign: "center", padding: 60 }}>Chargement...</div>;
  }

  const pointCount = mode === "centerline" ? points.length : leftPoints.length + rightPoints.length;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--color-bg)" }}>
      {/* Top Bar */}
      <div style={{
        height: 48, background: "var(--color-surface)", borderBottom: "1px solid var(--color-border)",
        display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 16px", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <button className="btn btn-secondary" style={{ padding: "4px 12px", fontSize: 12 }}
            onClick={() => navigate(`/session/${sessionName}`)}>
            ← Retour
          </button>
          <span style={{ fontSize: 14, fontWeight: 600 }}>🤿 {sessionName}</span>
          <span style={{ fontSize: 13, color: "var(--color-text-dim)" }}>
            {currentIdx + 1} / {session.images.length}
          </span>
          <span className="badge badge-cyan">{pointCount} points</span>
          {isRevision && <span className="badge badge-blue">Mode révision</span>}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {/* Mode toggle */}
          <div style={{
            display: "flex", border: "1px solid var(--color-border)", borderRadius: 6, overflow: "hidden",
          }}>
            <button style={{
              padding: "4px 12px", fontSize: 11, border: "none", cursor: "pointer",
              background: mode === "centerline" ? "var(--color-accent)" : "var(--color-surface)",
              color: mode === "centerline" ? "#0D1117" : "var(--color-text-dim)",
              fontWeight: 500,
            }} onClick={() => setMode("centerline")}>
              Ligne centrale
            </button>
            <button style={{
              padding: "4px 12px", fontSize: 11, border: "none", cursor: "pointer",
              background: mode === "contour" ? "var(--color-accent)" : "var(--color-surface)",
              color: mode === "contour" ? "#0D1117" : "var(--color-text-dim)",
              fontWeight: 500,
            }} onClick={() => setMode("contour")}>
              Contour câble
            </button>
          </div>
          {mode === "contour" && (
            <div style={{ display: "flex", border: "1px solid var(--color-border)", borderRadius: 6, overflow: "hidden" }}>
              <button style={{
                padding: "4px 10px", fontSize: 11, border: "none", cursor: "pointer",
                background: activeSide === "left" ? "#00FFFF" : "var(--color-surface)",
                color: activeSide === "left" ? "#0D1117" : "var(--color-text-dim)",
              }} onClick={() => setActiveSide("left")}>
                Gauche ({leftPoints.length})
              </button>
              <button style={{
                padding: "4px 10px", fontSize: 11, border: "none", cursor: "pointer",
                background: activeSide === "right" ? "#FFA500" : "var(--color-surface)",
                color: activeSide === "right" ? "#0D1117" : "var(--color-text-dim)",
              }} onClick={() => setActiveSide("right")}>
                Droite ({rightPoints.length})
              </button>
            </div>
          )}
          <button className="shortcut-key" style={{ cursor: "pointer" }} onClick={() => setShowShortcuts(true)}>?</button>
        </div>
      </div>

      {isRevision && (
        <div style={{
          background: "rgba(88, 166, 255, 0.1)", borderBottom: "1px solid rgba(88, 166, 255, 0.3)",
          padding: "6px 16px", fontSize: 13, color: "var(--color-blue)", textAlign: "center",
        }}>
          🔄 Mode révision — Cette image a déjà été annotée. Modifiez et mettez à jour.
        </div>
      )}

      {/* Main layout */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Left: thumbnails */}
        <div style={{
          width: 120, background: "var(--color-surface)", borderRight: "1px solid var(--color-border)",
          overflowY: "auto", flexShrink: 0, padding: 4,
        }}>
          {session.images.map((img, i) => (
            <div
              key={img.filename}
              onClick={() => goTo(i)}
              style={{
                position: "relative", cursor: "pointer", borderRadius: 6, overflow: "hidden",
                marginBottom: 4, border: i === currentIdx ? "2px solid var(--color-accent)" : "2px solid transparent",
                opacity: i === currentIdx ? 1 : 0.7,
                transition: "all 0.15s",
              }}
            >
              <img
                src={imageUrl(sessionName, img.filename)}
                alt=""
                style={{ width: "100%", aspectRatio: "16/9", objectFit: "cover", display: "block" }}
                loading="lazy"
              />
              <div style={{
                position: "absolute", top: 3, right: 3, width: 8, height: 8, borderRadius: "50%",
                background: img.status === "annotated" ? "var(--color-green)" :
                  img.status === "ignored" ? "var(--color-orange)" : "var(--color-text-dim)",
                border: "1.5px solid rgba(0,0,0,0.5)",
              }} />
            </div>
          ))}
        </div>

        {/* Center: canvas */}
        <div ref={containerRef} style={{ flex: 1, position: "relative", overflow: "hidden", cursor: isPanning ? "grabbing" : "crosshair" }}>
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            onContextMenu={handleContextMenu}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            style={{ width: "100%", height: "100%", display: "block" }}
          />
          {/* Tooltip */}
          <div style={{
            position: "absolute", bottom: 12, left: "50%", transform: "translateX(-50%)",
            background: "rgba(0,0,0,0.8)", padding: "6px 14px", borderRadius: 8, fontSize: 12,
            color: "var(--color-text-dim)", pointerEvents: "none", whiteSpace: "nowrap",
          }}>
            Clic = ajouter point · Clic droit = annuler · Entrée = confirmer · Molette = zoom
          </div>
          {/* Zoom indicator */}
          <div style={{
            position: "absolute", bottom: 12, left: 12, display: "flex", gap: 4, alignItems: "center",
          }}>
            <span style={{ fontSize: 11, color: "var(--color-text-dim)", background: "rgba(0,0,0,0.6)", padding: "2px 8px", borderRadius: 4 }}>
              {Math.round(zoom * 100)}%
            </span>
          </div>
          {/* Zoom buttons */}
          <div style={{ position: "absolute", bottom: 12, right: 12, display: "flex", gap: 4 }}>
            <button className="btn btn-secondary" style={{ padding: "2px 8px", fontSize: 14 }}
              onClick={() => setZoom((z) => Math.min(10, z * 1.2))}>+</button>
            <button className="btn btn-secondary" style={{ padding: "2px 8px", fontSize: 14 }}
              onClick={() => setZoom((z) => Math.max(0.2, z / 1.2))}>−</button>
            <button className="btn btn-secondary" style={{ padding: "2px 8px", fontSize: 11 }}
              onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>Reset</button>
          </div>
        </div>

        {/* Right: conditions panel */}
        <div style={{
          width: 320, background: "var(--color-surface)", borderLeft: "1px solid var(--color-border)",
          overflowY: "auto", flexShrink: 0, padding: 16,
        }}>
          <ConditionsPanel
            conditions={conditions}
            onChange={setConditions}
            onCopyPrevious={handleCopyConditions}
            copiedFeedback={copiedFeedback}
          />

          <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 16 }}>
            <button className="btn btn-green" style={{ width: "100%", justifyContent: "center" }} onClick={handleSave}>
              {isRevision ? "🔄 Mettre à jour" : "✅ Confirmer annotation"}
            </button>
            <button className="btn btn-orange" style={{ width: "100%", justifyContent: "center" }} onClick={handleIgnore}>
              ⏭ Ignorer
            </button>
            <button className="btn btn-red" style={{ width: "100%", justifyContent: "center", fontSize: 12 }} onClick={handleDeleteImg}>
              🗑 Supprimer l'image
            </button>
          </div>
        </div>
      </div>

      {/* Toast */}
      {toast && <div className="toast toast-success">{toast}</div>}

      {/* Shortcuts */}
      {showShortcuts && <ShortcutsPanel onClose={() => setShowShortcuts(false)} />}
    </div>
  );
}
