#!/usr/bin/env python3
"""Parse LDSC h2 logs for Omni-fMRI embeddings.

Inputs:
  - LDSC .log files from submit_ldsc_h2.pbs.

Outputs:
  - Per-embedding h2 summary TSV.
  - Model-level metric summary TSV.
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Omni-fMRI LDSC h2 logs.")
    parser.add_argument("--logs-glob", required=True, help="Glob for LDSC *.log files.")
    parser.add_argument("--output", required=True, help="Per-embedding h2 summary TSV.")
    parser.add_argument("--model-summary", default=None, help="Optional model-level summary TSV.")
    parser.add_argument("--embedding-regex", default=r"(emb_\d{3})", help="Regex extracting embedding_id from path.")
    return parser.parse_args()


def parse_estimate(text: str, label: str) -> tuple[float, float]:
    pattern = re.compile(rf"{re.escape(label)}:\s*([-+0-9.eE]+)\s*\(([-+0-9.eE]+)\)")
    match = pattern.search(text)
    if not match:
        return math.nan, math.nan
    return float(match.group(1)), float(match.group(2))


def parse_scalar(text: str, label: str) -> float:
    pattern = re.compile(rf"{re.escape(label)}:\s*([-+0-9.eE]+)")
    match = pattern.search(text)
    return float(match.group(1)) if match else math.nan


def parse_log(path: Path, embedding_regex: re.Pattern[str]) -> dict[str, object]:
    match = embedding_regex.search(str(path))
    if not match:
        raise ValueError(f"Could not extract embedding_id from path: {path}")
    embedding_id = match.group(1)
    text = path.read_text(encoding="utf-8", errors="replace")

    h2, h2_se = parse_estimate(text, "Total Observed scale h2")
    intercept, intercept_se = parse_estimate(text, "Intercept")
    lambda_gc = parse_scalar(text, "Lambda GC")
    mean_chi2 = parse_scalar(text, "Mean Chi^2")
    ratio, ratio_se = parse_estimate(text, "Ratio")
    if math.isnan(ratio):
        ratio = parse_scalar(text, "Ratio")
        ratio_se = math.nan

    status = "parsed" if math.isfinite(h2) else "missing_h2"
    return {
        "embedding_id": embedding_id,
        "h2": h2,
        "h2_se": h2_se,
        "intercept": intercept,
        "intercept_se": intercept_se,
        "lambda_gc": lambda_gc,
        "mean_chi2": mean_chi2,
        "ratio": ratio,
        "ratio_se": ratio_se,
        "status": status,
        "log_path": str(path),
    }


def metric_row(metric: str, value: object) -> dict[str, object]:
    return {"metric": metric, "value": value}


def main() -> int:
    args = parse_args()
    import pandas as pd

    paths = [Path(path) for path in sorted(glob.glob(args.logs_glob))]
    if not paths:
        raise FileNotFoundError(f"No logs matched --logs-glob {args.logs_glob!r}")
    regex = re.compile(args.embedding_regex)
    rows = [parse_log(path, regex) for path in paths]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values("embedding_id")
    table.to_csv(output, sep="\t", index=False, na_rep="NA")

    model_summary = Path(args.model_summary) if args.model_summary else output.with_suffix(".model_summary.tsv")
    parsed = table[table["status"] == "parsed"].copy()
    h2 = pd.to_numeric(parsed["h2"], errors="coerce")
    intercept = pd.to_numeric(parsed["intercept"], errors="coerce")
    summary_rows = [
        metric_row("ldsc_logs_found", len(paths)),
        metric_row("parsed_embeddings", int(h2.notna().sum())),
        metric_row("mean_h2", h2.mean(skipna=True)),
        metric_row("min_h2", h2.min(skipna=True)),
        metric_row("max_h2", h2.max(skipna=True)),
        metric_row("mean_intercept", intercept.mean(skipna=True)),
        metric_row("min_intercept", intercept.min(skipna=True)),
        metric_row("max_intercept", intercept.max(skipna=True)),
    ]
    pd.DataFrame(summary_rows).to_csv(model_summary, sep="\t", index=False, na_rep="NA")

    print(f"Wrote {output}")
    print(f"Wrote {model_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
