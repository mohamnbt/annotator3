#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "🚀 Démarrage COSMER Annotator v2..."
cd backend && uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ../frontend && npm run dev &
FRONTEND_PID=$!
trap "echo 'Arrêt...' && kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT INT TERM
echo "✅ Backend  : http://localhost:8000"
echo "✅ Frontend : http://localhost:5173"
wait
