#!/bin/sh
# entrypoint: ensure model exists then start uvicorn
if [ ! -f /app/models/baseline.pkl ]; then
  echo "Model not found at /app/models/baseline.pkl — you must mount or copy a model into /app/models"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8080
