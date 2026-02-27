#!/bin/sh
set -e


echo "  Check Me — Starting up"


if [ ! -f /model/artefacts/pipeline.pkl ]; then
  echo "[1/2] WARNING: No artefacts found — training fallback..."
  echo "      Run train_and_export.py locally to avoid this."
  python -m model.train --out /model/artefacts
else
  echo "[1/2] Pre-trained model loaded. ✓  (trained locally, bundled in image)"
fi

echo "[2/2] Starting FastAPI server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
