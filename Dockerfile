FROM python:3.11-slim

# required system deps for building some Python packages (kept minimal)
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

# ensure entrypoint is executable
RUN chmod +x /app/entrypoint.sh

# use a non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

ENV PORT=8080
EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
