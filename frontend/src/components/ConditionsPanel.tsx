import type { Conditions } from "../lib/api";

interface Props {
  conditions: Conditions;
  onChange: (c: Conditions) => void;
  onCopyPrevious: () => void;
  copiedFeedback: boolean;
}

const CURRENT_DIRECTIONS = ["—", "Unidirectionnel", "Bidirectionnel", "Turbulent"];
const CAMERA_ANGLES = ["—", "Face", "Latéral gauche", "Latéral droit", "Dessus", "Oblique 45°", "Autre"];
const TURBIDITIES = ["—", "Claire", "Légèrement turbide", "Turbide"];
const LIGHTINGS = ["—", "Lumière naturelle", "Lumière artificielle", "Faible luminosité", "Mixte"];

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <label style={{ display: "block", fontSize: 11, color: "var(--color-text-dim)", marginBottom: 3, fontWeight: 500 }}>
        {label}
      </label>
      {children}
    </div>
  );
}

export default function ConditionsPanel({ conditions, onChange, onCopyPrevious, copiedFeedback }: Props) {
  const set = (key: keyof Conditions, value: any) => {
    onChange({ ...conditions, [key]: value });
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h3 style={{ fontSize: 14, fontWeight: 600 }}>Conditions expérimentales</h3>
      </div>

      <button
        className="btn btn-secondary"
        style={{ width: "100%", justifyContent: "center", marginBottom: 12, fontSize: 12 }}
        onClick={onCopyPrevious}
      >
        {copiedFeedback ? "Conditions copiées ✓" : "📋 Copier conditions précédentes"}
      </button>

      <Field label="Annotateur">
        <input
          type="text"
          value={conditions.annotator_name || ""}
          onChange={(e) => set("annotator_name", e.target.value)}
          placeholder="Votre nom"
          style={{ fontSize: 12, padding: "6px 10px" }}
        />
      </Field>

      <Field label="Vitesse courant (cm/s)">
        <input
          type="number"
          value={conditions.current_speed_cm_s ?? ""}
          onChange={(e) => set("current_speed_cm_s", e.target.value ? parseFloat(e.target.value) : "")}
          min={0} max={200} step={0.5}
          style={{ fontSize: 12, padding: "6px 10px" }}
        />
      </Field>

      <Field label="Direction courant">
        <select
          value={conditions.current_direction || "—"}
          onChange={(e) => set("current_direction", e.target.value)}
          style={{ fontSize: 12, padding: "6px 10px" }}
        >
          {CURRENT_DIRECTIONS.map((d) => <option key={d} value={d}>{d}</option>)}
        </select>
      </Field>

      <Field label="Amplitude houle (cm)">
        <input
          type="number"
          value={conditions.wave_amplitude_cm ?? ""}
          onChange={(e) => set("wave_amplitude_cm", e.target.value ? parseFloat(e.target.value) : "")}
          min={0} max={100}
          style={{ fontSize: 12, padding: "6px 10px" }}
        />
      </Field>

      <Field label="Fréquence houle (Hz)">
        <input
          type="number"
          value={conditions.wave_frequency_hz ?? ""}
          onChange={(e) => set("wave_frequency_hz", e.target.value ? parseFloat(e.target.value) : "")}
          min={0} max={5} step={0.1}
          style={{ fontSize: 12, padding: "6px 10px" }}
        />
      </Field>

      <Field label="Vitesse vent (m/s)">
        <input
          type="number"
          value={conditions.wind_speed_m_s ?? ""}
          onChange={(e) => set("wind_speed_m_s", e.target.value ? parseFloat(e.target.value) : "")}
          min={0} max={30}
          style={{ fontSize: 12, padding: "6px 10px" }}
        />
      </Field>

      <Field label="Angle caméra">
        <select
          value={conditions.camera_angle || "—"}
          onChange={(e) => set("camera_angle", e.target.value)}
          style={{ fontSize: 12, padding: "6px 10px" }}
        >
          {CAMERA_ANGLES.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
      </Field>

      <Field label="Turbidité eau">
        <select
          value={conditions.water_turbidity || "—"}
          onChange={(e) => set("water_turbidity", e.target.value)}
          style={{ fontSize: 12, padding: "6px 10px" }}
        >
          {TURBIDITIES.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
      </Field>

      <Field label="Éclairage">
        <select
          value={conditions.lighting_condition || "—"}
          onChange={(e) => set("lighting_condition", e.target.value)}
          style={{ fontSize: 12, padding: "6px 10px" }}
        >
          {LIGHTINGS.map((l) => <option key={l} value={l}>{l}</option>)}
        </select>
      </Field>

      <Field label="Longueur corde immergée (cm)">
        <input
          type="number"
          value={conditions.immersed_length_cm ?? ""}
          onChange={(e) => set("immersed_length_cm", e.target.value ? parseFloat(e.target.value) : "")}
          min={0}
          style={{ fontSize: 12, padding: "6px 10px" }}
        />
      </Field>

      <Field label="Hauteur bouée→surface (cm)">
        <input
          type="number"
          value={conditions.buoy_to_surface_cm ?? ""}
          onChange={(e) => set("buoy_to_surface_cm", e.target.value ? parseFloat(e.target.value) : "")}
          min={0}
          style={{ fontSize: 12, padding: "6px 10px" }}
        />
      </Field>

      <Field label="Hauteur d'eau canal (cm)">
        <input
          type="number"
          value={conditions.canal_water_depth_cm ?? ""}
          onChange={(e) => set("canal_water_depth_cm", e.target.value ? parseFloat(e.target.value) : "")}
          min={0}
          style={{ fontSize: 12, padding: "6px 10px" }}
        />
      </Field>

      <Field label="Notes">
        <textarea
          value={conditions.notes || ""}
          onChange={(e) => set("notes", e.target.value)}
          rows={2}
          placeholder="Notes libres..."
          style={{ fontSize: 12, padding: "6px 10px" }}
        />
      </Field>
    </div>
  );
}
