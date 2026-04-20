#!/usr/bin/env bash
set -e

# If a persistent volume is mounted at /data (or DATA_ROOT), symlink the app's
# runtime directories there so uploads and saved files survive deploys.
# If /data isn't mounted or isn't writable, skip silently — the app will use
# its local directories inside the container (fine for ephemeral deploys).
DATA_ROOT="${DATA_ROOT:-/data}"

if mkdir -p "${DATA_ROOT}" 2>/dev/null && [ -w "${DATA_ROOT}" ]; then
  for dir in uploads deck_contexts saved_extractions saved_reports; do
    target="${DATA_ROOT}/${dir}"
    link="/app/${dir}"
    mkdir -p "$target"
    if [ ! -L "$link" ] && [ ! -e "$link" ]; then
      ln -s "$target" "$link"
    elif [ -d "$link" ] && [ ! -L "$link" ]; then
      # Directory exists but isn't a symlink — migrate contents then replace
      cp -r "${link}/." "${target}/" 2>/dev/null || true
      rm -rf "$link"
      ln -s "$target" "$link"
    fi
  done
fi

exec "$@"
