import { Route, Switch } from "wouter";
import HomePage from "./pages/HomePage";
import SessionPage from "./pages/SessionPage";
import AnnotatorPage from "./pages/AnnotatorPage";

function Header() {
  return (
    <header
      style={{
        background: "var(--color-surface)",
        borderBottom: "1px solid var(--color-border)",
        padding: "0 24px",
        height: 56,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}
    >
      <a
        href="/"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          textDecoration: "none",
          color: "var(--color-text)",
        }}
      >
        <span style={{ fontSize: 24 }}>🤿</span>
        <span
          style={{
            fontSize: 18,
            fontWeight: 700,
            letterSpacing: "-0.5px",
            background: "linear-gradient(135deg, var(--color-accent), var(--color-green))",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          COSMER Annotator
        </span>
      </a>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span
          style={{
            fontSize: 11,
            color: "var(--color-text-dim)",
            padding: "4px 8px",
            background: "var(--color-bg)",
            borderRadius: 6,
            border: "1px solid var(--color-border)",
          }}
        >
          Laboratoire COSMER — Université de Toulon
        </span>
        <span
          className="shortcut-key"
          style={{ cursor: "pointer", fontSize: 14 }}
          title="Raccourcis clavier"
        >
          ?
        </span>
      </div>
    </header>
  );
}

export default function App() {
  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Route path="/session/:name/annotate">
        {(params) => <AnnotatorPage sessionName={params.name} />}
      </Route>
      <Switch>
        <Route path="/session/:name/annotate">
          {() => null}
        </Route>
        <Route path="/">
          {() => (
            <>
              <Header />
              <main style={{ flex: 1, padding: "24px 32px", maxWidth: 1400, margin: "0 auto", width: "100%" }}>
                <HomePage />
              </main>
            </>
          )}
        </Route>
        <Route path="/session/:name">
          {(params) => (
            <>
              <Header />
              <main style={{ flex: 1, padding: "24px 32px", maxWidth: 1400, margin: "0 auto", width: "100%" }}>
                <SessionPage sessionName={params.name} />
              </main>
            </>
          )}
        </Route>
      </Switch>
    </div>
  );
}
