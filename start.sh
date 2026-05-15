#!/bin/bash
set -e

# Dossier racine du projet (là où se trouve ce script)
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "🚀 Démarrage COSMER Annotator v2..."
echo "📁 Racine : $ROOT"

# Vérifications
if [ ! -d "$ROOT/backend" ]; then
  echo "❌ Dossier backend introuvable dans $ROOT"
  exit 1
fi
if [ ! -d "$ROOT/frontend" ]; then
  echo "❌ Dossier frontend introuvable dans $ROOT"
  echo "   Lancez d'abord : cd frontend && npm install"
  exit 1
fi

# Backend
cd "$ROOT/backend"
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# Frontend
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

trap "echo '🛑 Arrêt...' && kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM

echo "✅ Backend  : http://localhost:8000"
echo "✅ Frontend : http://localhost:5173"
echo "   Appuyez sur Ctrl+C pour stopper."

wait
