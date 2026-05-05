interface Props {
  onClose: () => void;
}

const SHORTCUTS = [
  { key: "Clic gauche", desc: "Ajouter un point" },
  { key: "Clic droit", desc: "Annuler dernier point" },
  { key: "Z", desc: "Annuler dernier point" },
  { key: "Escape", desc: "Effacer tous les points" },
  { key: "Enter", desc: "Confirmer l'annotation" },
  { key: "←", desc: "Image précédente" },
  { key: "→", desc: "Image suivante" },
  { key: "Molette", desc: "Zoom avant/arrière" },
  { key: "Espace + glisser", desc: "Panoramique (pan)" },
  { key: "Clic molette + glisser", desc: "Panoramique (pan)" },
  { key: "?", desc: "Afficher/masquer ce panneau" },
];

export default function ShortcutsPanel({ onClose }: Props) {
  return (
    <div className="shortcuts-panel" onClick={onClose}>
      <div className="shortcuts-content animate-fade-in-scale" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600 }}>⌨️ Raccourcis clavier</h2>
          <button
            className="btn btn-secondary"
            style={{ padding: "4px 12px", fontSize: 12 }}
            onClick={onClose}
          >
            Fermer
          </button>
        </div>
        {SHORTCUTS.map((s) => (
          <div key={s.key} className="shortcut-row">
            <span style={{ fontSize: 13 }}>{s.desc}</span>
            <span className="shortcut-key">{s.key}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
