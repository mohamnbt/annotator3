#!/bin/bash
cd backend && uvicorn main:app --reload --port 8000 &
cd frontend && npm run dev
wait
