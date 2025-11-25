FROM python:3.11-slim

# minimal system deps for building packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      gfortran \
      libffi-dev \
      libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# copy app code
COPY . /app

# ensure python sees the package
ENV PYTHONPATH=/app

# create a non-root user and use it
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

ENV PORT=8080
EXPOSE 8080

# start uvicorn directly (no entrypoint.sh to avoid CRLF/executable issues)
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
