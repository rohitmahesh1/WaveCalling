# ---------- Frontend build ----------
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend

# Install deps and build
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ .
RUN npm run build

# ---------- Backend runtime ----------
FROM python:3.12-slim AS backend

WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY src/ ./src

# Copy frontend build into backend's web dir
COPY --from=frontend-build /app/frontend/dist ./web

# Ensure FastAPI can find the web dir (WEB_DIR=./web)
ENV WEB_DIR=/app/web
ENV PYTHONUNBUFFERED=1

EXPOSE 800
CMD ["python", "-m", "src.service.api"]
