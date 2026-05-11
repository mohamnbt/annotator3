#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Trouve uvicorn dans conda ou venv ──────────────────────────────
if command -v uvicorn &>/dev/null; then
    UVICORN="uvicorn"
elif [ -f "$ROOT/backend/.venv/bin/uvicorn" ]; then
    UVICORN="$ROOT/backend/.venv/bin/uvicorn"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/uvicorn" ]; then
    UVICORN="$CONDA_PREFIX/bin/uvicorn"
else
    # Cherche dans tous les envs conda connus
    UVICORN=$(find ~/opt/anaconda3 ~/anaconda3 ~/miniconda3 ~/miniforge3 \
               -name uvicorn -type f 2>/dev/null | head -1)
    if [ -z "$UVICORN" ]; then
        echo "❌  uvicorn introuvable."
        echo "   Lance d'abord : conda activate <ton_env>  ou  source backend/.venv/bin/activate"
        exit 1
    fi
fi

echo "✅  uvicorn : $UVICORN"

# ── Backend ────────────────────────────────────────────────────────
cd "$ROOT/backend"
"$UVICORN" main:app --reload --port 8000 &
BACKEND_PID=$!
echo "🚀  Backend PID $BACKEND_PID  →  http://localhost:8000"

# ── Frontend ───────────────────────────────────────────────────────
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!
echo "🎨  Frontend PID $FRONTEND_PID  →  http://localhost:5173"

# ── Arrêt propre sur Ctrl+C ────────────────────────────────────────
trap "echo '\n🛑  Arrêt...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
