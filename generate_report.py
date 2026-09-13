"""
Re9lay - Progress Report Generator (CLI entry point)

Usage:
    python generate_report.py session.csv --out report.pdf --emg-threshold 400

If --session-label is not given, it's parsed automatically from the CSV
filename when named like yyyy_mm_dd_hh_mm.csv (e.g. 2026_09_04_04_43.csv).
Falls back to "now" if the filename doesn't match.

Reads a Unity-logged session CSV (20Hz), computes EMG/motion/score
metrics, generates charts, and assembles a PDF progress report.
This script is standalone - run it after a session, separately from Unity.
"""

import argparse
import os
import re
import tempfile
from datetime import datetime
from typing import Optional

from metrics import load_session, compute_all_metrics
from charts import generate_all_charts
from report import build_report

FILENAME_PATTERN = re.compile(
    r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})"
)


def session_label_from_filename(csv_path: str) -> Optional[str]:
    """Parse yyyy_mm_dd_hh_mm from the CSV filename, if present."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    match = FILENAME_PATTERN.search(stem)
    if not match:
        return None
    year, month, day, hour, minute = match.groups()
    try:
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate Re9lay progress report PDF")
    parser.add_argument("csv_path", help="Path to session CSV")
    parser.add_argument("--out", default="progress_report.pdf", help="Output PDF path")
    parser.add_argument("--emg-threshold", type=float, default=400.0,
                         help="EMG value threshold for contraction detection")
    parser.add_argument("--session-label", default=None,
                         help="Label shown on report (auto-parsed from filename if omitted)")
    parser.add_argument("--logo", default=None, help="Path to logo image (png/jpg) for the header")
    parser.add_argument("--rest-baseline", type=float, default=None,
                         help="EMG rest baseline from calibration (enables %%MVC reporting)")
    parser.add_argument("--max-contraction", type=float, default=None,
                         help="EMG max contraction from calibration (enables %%MVC reporting)")
    args = parser.parse_args()

    session_label = args.session_label or session_label_from_filename(args.csv_path)

    df = load_session(args.csv_path)
    metrics = compute_all_metrics(
        df, emg_threshold=args.emg_threshold,
        rest_baseline=args.rest_baseline, max_contraction=args.max_contraction,
    )

    with tempfile.TemporaryDirectory() as chart_dir:
        chart_paths = generate_all_charts(df, metrics, chart_dir, emg_threshold=args.emg_threshold)
        build_report(metrics, chart_paths, args.out, session_label=session_label,
                     logo_path=args.logo)

    print(f"Report written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
