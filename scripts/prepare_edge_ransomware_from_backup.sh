#!/usr/bin/env bash
set -euo pipefail

# Prepares an Edge-IIoTSet ransomware raw dataset directory from a backup folder.
#
# Backup folder (created by scripts/sync_edge_ransomware_csvs.sh) contains:
#   Benign Traffic.csv
#   Ransomware.csv
#
# The project loader expects:
#   <DEST_BASE>/edge_ransomware/raw/Benign%20Traffic.csv
#   <DEST_BASE>/edge_ransomware/raw/Ransomware.csv
#
# We create symlinks (not copies) by default, to avoid duplicating ~1GB files.
#
# Usage:
#   ./scripts/prepare_edge_ransomware_from_backup.sh --backup-dir data/_backups_edge_ransomware/20260204-105659
#   ./scripts/prepare_edge_ransomware_from_backup.sh --backup-dir ... --dest-base ./data/_tmp_edge_backup_ids_ransomware
#   ./scripts/prepare_edge_ransomware_from_backup.sh --backup-dir ... --copy

BACKUP_DIR=""
DEST_BASE="./data/_tmp_edge_backup_ids_ransomware"
DO_COPY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backup-dir)
      BACKUP_DIR="${2:-}"
      shift 2
      ;;
    --dest-base)
      DEST_BASE="${2:-}"
      shift 2
      ;;
    --copy)
      DO_COPY="true"
      shift
      ;;
    -h|--help)
      sed -n '1,140p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BACKUP_DIR" ]]; then
  echo "Error: --backup-dir is required" >&2
  exit 2
fi

SRC_BENIGN="${BACKUP_DIR}/Benign Traffic.csv"
SRC_RANSOM="${BACKUP_DIR}/Ransomware.csv"

if [[ ! -f "$SRC_BENIGN" || ! -f "$SRC_RANSOM" ]]; then
  echo "Error: backup dir does not contain expected files:" >&2
  echo "  $SRC_BENIGN" >&2
  echo "  $SRC_RANSOM" >&2
  exit 1
fi

RAW_DIR="${DEST_BASE}/edge_ransomware/raw"
mkdir -p "$RAW_DIR"

DEST_BENIGN="${RAW_DIR}/Benign%20Traffic.csv"
DEST_RANSOM="${RAW_DIR}/Ransomware.csv"

rm -f "$DEST_BENIGN" "$DEST_RANSOM"

if [[ "$DO_COPY" == "true" ]]; then
  echo "Copying backup -> $RAW_DIR"
  cp -f "$SRC_BENIGN" "$DEST_BENIGN"
  cp -f "$SRC_RANSOM" "$DEST_RANSOM"
else
  echo "Symlinking backup -> $RAW_DIR"
  ln -s "$(realpath "$SRC_BENIGN")" "$DEST_BENIGN"
  ln -s "$(realpath "$SRC_RANSOM")" "$DEST_RANSOM"
fi

echo
echo "Prepared:"
ls -lh "$DEST_BENIGN" "$DEST_RANSOM"
echo
echo "Use with config data_base_path: ${DEST_BASE}/"

