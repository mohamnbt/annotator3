import { Route, Switch } from "wouter";
import HomePage from "./pages/HomePage";
import SessionPage from "./pages/SessionPage";
import AnnotatorPage from "./pages/AnnotatorPage";
import GlobalTrainPage from "./pages/GlobalTrainPage";

function Header() {
  return (
    <header className="bg-surface border-b border-border px-6 h-14 flex items-center justify-between sticky top-0 z-[100]">
      <a href="/" className="flex items-center gap-3 no-underline group">
        <span className="text-2xl group-hover:scale-110 transition-transform">🤿</span>
        <span className="text-xl font-black tracking-tighter bg-gradient-to-r from-accent to-green bg-clip-text text-transparent">
          COSMER Annotator
        </span>
      </a>
      <div className="flex items-center gap-4">
        <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-dim px-3 py-1 bg-bg rounded-full border border-border">
          Laboratoire COSMER — Université de Toulon
        </span>
      </div>
    </header>
  );
}

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Route path="/session/:name/annotate">
        {(params) => <AnnotatorPage sessionName={params.name} />}
      </Route>
      <Switch>
        <Route path="/session/:name/annotate">{() => null}</Route>
        <Route path="/">
          {() => (
            <><Header /><main className="flex-1 px-8 py-8 max-w-[1600px] mx-auto w-full"><HomePage /></main></>
          )}
        </Route>
        <Route path="/session/:name">
          {(params) => (
            <><Header /><main className="flex-1 px-8 py-8 max-w-[1600px] mx-auto w-full"><SessionPage sessionName={params.name} /></main></>
          )}
        </Route>
        <Route path="/train/global">
          {() => (
            <><Header /><main className="flex-1 px-8 py-8 max-w-[1600px] mx-auto w-full"><GlobalTrainPage /></main></>
          )}
        </Route>
      </Switch>
    </div>
  );
}
