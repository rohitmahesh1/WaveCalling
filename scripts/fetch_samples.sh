#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/fetch_samples.sh [--repo owner/repo] [--dir path] [--force]
# Examples:
#   ./scripts/fetch_samples.sh
#   ./scripts/fetch_samples.sh --force
#   ./scripts/fetch_samples.sh --repo rohitmahesh1/WaveCalling --dir samples

REPO="rohitmahesh1/WaveCalling"
DIR="samples"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO="$2"; shift 2 ;;
    --dir)  DIR="$2";  shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if ! command -v gh >/dev/null 2>&1; then
  echo "Error: GitHub CLI (gh) not found. Install from https://cli.github.com/"; exit 1
fi

mkdir -p "$DIR"

echo "Checking latest release for $REPO..."
TAG="$(gh release view --repo "$REPO" --json tagName -q '.tagName' || true)"
ASSETS="$(gh release view --repo "$REPO" --json assets -q '.assets[].name' || true)"

if [[ -z "${TAG:-}" ]]; then
  echo "No release found for $REPO (or not accessible)."; exit 1
fi
if [[ -z "${ASSETS//[$'\t\r\n ']/}" ]]; then
  echo "No assets found on release $TAG."; exit 1
fi

echo "Latest release: $TAG"
echo "Assets:"
printf '  - %s\n' $ASSETS

# Download only missing files unless --force
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  if [[ $FORCE -eq 0 && -s "$DIR/$name" ]]; then
    echo "✓ $name already exists; skipping"
    continue
  fi
  echo "↓ Downloading $name"
  gh release download \
    --repo "$REPO" \
    --pattern "$name" \
    --dir "$DIR" \
    --clobber
done <<< "$ASSETS"

printf "%s\n" "$TAG" > "$DIR/.samples_release_tag"
echo "Samples are ready in '$DIR/'."
