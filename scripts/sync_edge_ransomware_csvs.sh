#!/usr/bin/env bash
set -euo pipefail

# Sync Edge-IIoTSet ransomware CSVs between the repo "raw" location and project root.
#
# Repo pipeline expects (raw):
#   data/ids_ransomware/edge_ransomware/raw/Benign%20Traffic.csv
#   data/ids_ransomware/edge_ransomware/raw/Ransomware.csv
#
# You currently also have (root):
#   ./Benign Traffic.csv
#   ./Ransomware.csv
#
# This script can make the ROOT files identical to RAW (recommended), or the inverse.
#
# Usage:
#   ./scripts/sync_edge_ransomware_csvs.sh --raw-to-root   # root becomes identical to raw
#   ./scripts/sync_edge_ransomware_csvs.sh --root-to-raw   # raw becomes identical to root
#
# Safety:
# - Creates timestamped backups before overwriting.

MODE="${1:-}"
if [[ "$MODE" != "--raw-to-root" && "$MODE" != "--root-to-raw" ]]; then
  echo "Usage: $0 --raw-to-root | --root-to-raw" >&2
  exit 2
fi

RAW_DIR="data/ids_ransomware/edge_ransomware/raw"
RAW_BENIGN="${RAW_DIR}/Benign%20Traffic.csv"
RAW_RANSOM="${RAW_DIR}/Ransomware.csv"

ROOT_BENIGN="Benign Traffic.csv"
ROOT_RANSOM="Ransomware.csv"

ts="$(date +%Y%m%d-%H%M%S)"
backup_dir="data/_backups_edge_ransomware/${ts}"
mkdir -p "$backup_dir"

backup_if_exists () {
  local p="$1"
  if [[ -f "$p" ]]; then
    cp -f "$p" "${backup_dir}/$(basename "$p")"
  fi
}

echo "Backup dir: $backup_dir"

if [[ "$MODE" == "--raw-to-root" ]]; then
  [[ -f "$RAW_BENIGN" ]] || { echo "Missing: $RAW_BENIGN" >&2; exit 1; }
  [[ -f "$RAW_RANSOM" ]] || { echo "Missing: $RAW_RANSOM" >&2; exit 1; }

  backup_if_exists "$ROOT_BENIGN"
  backup_if_exists "$ROOT_RANSOM"

  echo "Copying RAW -> ROOT..."
  cp -f "$RAW_BENIGN" "$ROOT_BENIGN"
  cp -f "$RAW_RANSOM" "$ROOT_RANSOM"
else
  [[ -f "$ROOT_BENIGN" ]] || { echo "Missing: $ROOT_BENIGN" >&2; exit 1; }
  [[ -f "$ROOT_RANSOM" ]] || { echo "Missing: $ROOT_RANSOM" >&2; exit 1; }

  mkdir -p "$RAW_DIR"
  backup_if_exists "$RAW_BENIGN"
  backup_if_exists "$RAW_RANSOM"

  echo "Copying ROOT -> RAW..."
  cp -f "$ROOT_BENIGN" "$RAW_BENIGN"
  cp -f "$ROOT_RANSOM" "$RAW_RANSOM"
fi

echo
echo "SHA256 after sync:"
sha256sum "$ROOT_BENIGN" "$RAW_BENIGN"
sha256sum "$ROOT_RANSOM" "$RAW_RANSOM"
echo
echo "Done."

