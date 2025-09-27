# ---------- Frontend build ----------
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend

# Install deps and build the SPA
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .
# The frontend build writes to /app/web (Vite outDir is ../web)
RUN npm run build

# ---------- Backend runtime ----------
FROM python:3.12-slim AS backend
WORKDIR /app

# Python runtime hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Samples mounted at runtime ----
RUN mkdir -p /app/samples
ENV SAMPLES_DIR=/app/samples

# Configs (read-only in compose)
COPY configs/ ./configs
ENV DEFAULT_CONFIG=/app/configs/default.yaml

# Backend code and models
COPY src/ ./src
COPY export/ ./export

# Frontend build output produced at /app/web by the previous stage
COPY --from=frontend-build /app/web ./web
ENV WEB_DIR=/app/web

EXPOSE 8080
CMD ["python", "-m", "src.service.api"]
