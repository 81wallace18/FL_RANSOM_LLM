#!/usr/bin/env python3
"""
Summarize Edge-IIoTSet ransomware datasets used in this repo.

Focus:
  - raw CSV integrity (rows, IPv4 ratio, 0.0.0.0 placeholders)
  - device diversity (unique Src IPs)
  - temporal overlap (timestamp ranges)
  - processed split sizes (train/calibration/test) when available

Examples:
  ./.venv/bin/python scripts/summarize_edge_dataset.py --dataset edge_ransomware_new --filter_ipv4_only --drop_zero_ips
  ./.venv/bin/python scripts/summarize_edge_dataset.py --dataset edge_ransomware
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


IPV4_RX = r"^(?:\d{1,3}\.){3}\d{1,3}$"
ZERO_IPS = {"0", "0.0.0.0", "::", ""}


@dataclass
class RawStats:
    rows: int
    ipv4_rows: int
    zero_rows: int
    unique_ipv4_src: int
    ts_min: str
    ts_max: str


def _parse_ts(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    missing = ts.isna()
    if missing.any():
        ts.loc[missing] = pd.to_datetime(series[missing], errors="coerce", dayfirst=True)
    return ts


def _raw_stats(csv_path: Path, *, filter_ipv4_only: bool, drop_zero_ips: bool, chunksize: int = 200_000) -> RawStats:
    rows = 0
    ipv4_rows = 0
    zero_rows = 0
    uniq_ipv4 = set()
    ts_min = None
    ts_max = None

    for chunk in pd.read_csv(
        csv_path,
        usecols=["Src IP", "Timestamp"],
        chunksize=chunksize,
        skipinitialspace=True,
        low_memory=False,
    ):
        src = chunk["Src IP"].astype("string").str.strip()
        rows += int(len(src))

        src_fill = src.fillna("")
        zero_rows += int(src_fill.isin(list(ZERO_IPS)).sum())

        ipv4_mask = src_fill.str.match(IPV4_RX)
        ipv4_rows += int(ipv4_mask.sum())

        # Apply optional filters to the uniqueness counting (match processor behavior)
        keep = src.notna()
        if drop_zero_ips:
            keep &= ~src_fill.isin(list(ZERO_IPS))
        if filter_ipv4_only:
            keep &= ipv4_mask

        uniq_ipv4.update(src[keep].unique().tolist())

        ts = _parse_ts(chunk["Timestamp"])
        if ts.notna().any():
            cmin = ts.min()
            cmax = ts.max()
            ts_min = cmin if ts_min is None or cmin < ts_min else ts_min
            ts_max = cmax if ts_max is None or cmax > ts_max else ts_max

    return RawStats(
        rows=rows,
        ipv4_rows=ipv4_rows,
        zero_rows=zero_rows,
        unique_ipv4_src=len(uniq_ipv4),
        ts_min=str(ts_min) if ts_min is not None else "NaT",
        ts_max=str(ts_max) if ts_max is not None else "NaT",
    )


def _processed_stats(processed_dir: Path) -> None:
    for split in ("train.csv", "calibration.csv", "test.csv"):
        path = processed_dir / split
        if not path.exists():
            print(f"- {split}: MISSING")
            continue

        df = pd.read_csv(path, usecols=["Src IP", "Label", "Timestamp"])
        rows = len(df)
        benign = int((df["Label"] == 0).sum()) if "Label" in df.columns else 0
        ransom = int((df["Label"] == 1).sum()) if "Label" in df.columns else 0
        uniq_b = int(df.loc[df["Label"] == 0, "Src IP"].astype(str).nunique()) if benign else 0
        uniq_r = int(df.loc[df["Label"] == 1, "Src IP"].astype(str).nunique()) if ransom else 0
        ts = _parse_ts(df["Timestamp"])
        print(
            f"- {split}: rows={rows} benign={benign} ransom={ransom} "
            f"uniq_src_benign={uniq_b} uniq_src_ransom={uniq_r} "
            f"ts_min={ts.min()} ts_max={ts.max()}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize Edge ransomware datasets (raw + processed).")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset folder name under data/ids_ransomware/")
    ap.add_argument("--data_base_path", type=str, default="./data/ids_ransomware", help="Base path for datasets.")
    ap.add_argument("--filter_ipv4_only", action="store_true", help="Apply IPv4-only filter (for unique Src IP counting).")
    ap.add_argument("--drop_zero_ips", action="store_true", help="Drop 0/0.0.0.0/:: placeholders (for unique Src IP counting).")
    args = ap.parse_args()

    base = Path(args.data_base_path) / args.dataset
    raw_dir = base / "raw"
    processed_dir = base / "processed"

    print(f"Dataset: {args.dataset}")
    print(f"Raw dir: {raw_dir}")
    print(f"Processed dir: {processed_dir}")
    print(f"Filters: filter_ipv4_only={args.filter_ipv4_only}, drop_zero_ips={args.drop_zero_ips}")

    benign_csv = raw_dir / "Benign%20Traffic.csv"
    ransom_csv = raw_dir / "Ransomware.csv"
    if not benign_csv.exists() or not ransom_csv.exists():
        raise SystemExit("Missing raw CSVs. Expected Benign%20Traffic.csv and Ransomware.csv under raw/.")

    print("\nRaw (Benign%20Traffic.csv):")
    b = _raw_stats(benign_csv, filter_ipv4_only=args.filter_ipv4_only, drop_zero_ips=args.drop_zero_ips)
    print(
        f"- rows={b.rows} ipv4_ratio={b.ipv4_rows / b.rows:.5f} zero_ratio={b.zero_rows / b.rows:.5f} "
        f"unique_src_after_filters={b.unique_ipv4_src} ts_min={b.ts_min} ts_max={b.ts_max}"
    )

    print("\nRaw (Ransomware.csv):")
    r = _raw_stats(ransom_csv, filter_ipv4_only=args.filter_ipv4_only, drop_zero_ips=args.drop_zero_ips, chunksize=50_000)
    print(
        f"- rows={r.rows} ipv4_ratio={r.ipv4_rows / r.rows:.5f} zero_ratio={r.zero_rows / r.rows:.5f} "
        f"unique_src_after_filters={r.unique_ipv4_src} ts_min={r.ts_min} ts_max={r.ts_max}"
    )

    if processed_dir.exists():
        print("\nProcessed splits:")
        _processed_stats(processed_dir)
    else:
        print("\nProcessed splits: directory missing (run preprocessing first).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

