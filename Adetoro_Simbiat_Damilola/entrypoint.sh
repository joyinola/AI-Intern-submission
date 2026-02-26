#!/bin/sh
set -e

echo "============================================"
echo "  Check Me — Starting up"
echo "============================================"

# Train the model if artefacts don't already exist
# (allows re-use of a mounted volume with pre-trained weights)
if [ ! -f /model/artefacts/pipeline.pkl ]; then
  echo "[1/2] Training model..."
  python model/train.py --out /model/artefacts
else
  echo "[1/2] Model artefacts already exist — skipping training."
fi

echo "[2/2] Starting FastAPI server..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
