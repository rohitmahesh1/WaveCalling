#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash ./scripts/fetch_models.sh [--repo owner/repo] [--dir path] [--force] [--tag vX.Y.Z] [--patterns "model_*.onnx *.onnx"]
# This script downloads ONLY model assets (.onnx) from a GitHub Release.

REPO="rohitmahesh1/WaveCalling"
DIR="export"
FORCE=0
TAG=""
MODEL_PATTERNS="model_*.onnx *.onnx"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)     REPO="$2"; shift 2 ;;
    --dir)      DIR="$2";  shift 2 ;;
    --force)    FORCE=1;   shift ;;
    --tag)      TAG="$2";  shift 2 ;;
    --patterns) MODEL_PATTERNS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) not found. Install from https://cli.github.com/"; exit 1
fi

mkdir -p "$DIR"

# Pick latest release tag unless --tag provided
if [[ -z "${TAG}" ]]; then
  echo "Checking latest release for $REPO..."
  TAG="$(gh release view --repo "$REPO" --json tagName -q '.tagName' || true)"
  [[ -z "${TAG}" ]] && { echo "No release found for $REPO."; exit 1; }
fi
echo "Using release tag: $TAG"

ASSETS="$(gh release view "$TAG" --repo "$REPO" --json assets -q '.assets[].name' || true)"
[[ -z "${ASSETS//[$'\t\r\n ']/}" ]] && { echo "No assets found on $TAG."; exit 1; }

echo "Selecting model assets…"
SELECTED=""
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  # match against any of the MODEL_PATTERNS
  match=0
  for pat in $MODEL_PATTERNS; do
    case "$name" in
      $pat) match=1; break;;
    esac
  done
  [[ $match -eq 1 ]] && SELECTED+="$name"$'\n'
done <<< "$ASSETS"

[[ -z "${SELECTED//[$'\t\r\n ']/}" ]] && { echo "No model assets matched patterns."; exit 1; }

echo "Models to download:"
printf '  - %s\n' $SELECTED

# Download only missing files unless --force
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  if [[ $FORCE -eq 0 && -s "$DIR/$name" ]]; then
    echo "✓ $name already exists; skipping"
    continue
  fi
  echo "↓ Downloading $name"
  gh release download "$TAG" \
    --repo "$REPO" \
    --pattern "$name" \
    --dir "$DIR" \
    --clobber
done <<< "$SELECTED"

printf "%s\n" "$TAG" > "$DIR/.models_release_tag"
echo "Models are ready in '$DIR/'."
