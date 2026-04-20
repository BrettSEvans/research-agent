#!/usr/bin/env bash
set -e

# Ensure persistent data directories exist under /data (Railway volume mount)
# and symlink them into /app so the application code finds them at expected paths.
DATA_ROOT="${DATA_ROOT:-/data}"

for dir in uploads deck_contexts saved_extractions saved_reports; do
  target="${DATA_ROOT}/${dir}"
  link="/app/${dir}"
  mkdir -p "$target"
  if [ ! -L "$link" ] && [ ! -e "$link" ]; then
    ln -s "$target" "$link"
  elif [ -d "$link" ] && [ ! -L "$link" ]; then
    # Directory exists but isn't a symlink — move contents then replace with symlink
    cp -r "${link}/." "${target}/" 2>/dev/null || true
    rm -rf "$link"
    ln -s "$target" "$link"
  fi
done

exec "$@"
