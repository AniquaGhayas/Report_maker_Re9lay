"""
Re9lay - Cloud report generation API.

Wraps the existing metrics.py / charts.py / report.py pipeline behind a
single HTTP endpoint so the Unity app can upload a session CSV and get
back a finished PDF, instead of needing a PC + manual script run.

The Re9lay logo is bundled in this repo (assets/logo.png) and used by
default on every report, so Unity does NOT need to upload it on every
request. If a 'logo' file IS uploaded with the request, it overrides
the bundled one for that report only.

Run locally:
    uvicorn server:app --host 0.0.0.0 --port 8000

Deploy: push this file + metrics.py/charts.py/report.py + requirements.txt
+ assets/logo.png to Render (or any host that runs a Python web service).
"""

import os
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from metrics import load_session, compute_all_metrics
from charts import generate_all_charts
from report import build_report

app = FastAPI(title="Re9lay Report API")

# Bundled logo, shipped in the repo. Update this file to change the
# logo used on every report without touching any code.
BUNDLED_LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-report")
async def generate_report(
    csv_file: UploadFile = File(...),
    session_label: str = Form(None),
    emg_threshold: float = Form(400.0),
    logo: UploadFile = File(None),
):
    """
    Accepts a session CSV (multipart form field 'csv_file'), an optional
    'session_label', an optional 'emg_threshold', and an optional 'logo'
    override image. Uses the bundled logo (assets/logo.png) unless a
    'logo' file is explicitly uploaded with this request. Returns the
    generated PDF as the response body.
    """
    if not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "csv_file must be a .csv file")

    work_dir = tempfile.mkdtemp(prefix="re9lay_")
    try:
        csv_path = os.path.join(work_dir, csv_file.filename)
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(csv_file.file, f)

        # Use an uploaded logo override if given, otherwise fall back
        # to the bundled logo shipped with this repo (if it exists).
        logo_path = None
        if logo is not None:
            logo_path = os.path.join(work_dir, logo.filename)
            with open(logo_path, "wb") as f:
                shutil.copyfileobj(logo.file, f)
        elif os.path.exists(BUNDLED_LOGO_PATH):
            logo_path = BUNDLED_LOGO_PATH

        label = session_label or os.path.splitext(csv_file.filename)[0]

        df = load_session(csv_path)
        metrics = compute_all_metrics(df, emg_threshold=emg_threshold)

        chart_dir = os.path.join(work_dir, "charts")
        chart_paths = generate_all_charts(df, metrics, chart_dir, emg_threshold=emg_threshold)

        out_path = os.path.join(work_dir, "report.pdf")
        build_report(metrics, chart_paths, out_path, session_label=label, logo_path=logo_path)

        return FileResponse(out_path, media_type="application/pdf", filename="progress_report.pdf")

    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(500, f"Report generation failed: {e}")
