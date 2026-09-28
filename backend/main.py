"""FastAPI backend: EEG-TCFNet home-appliance prediction service."""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.eeg_tcfnet import ApplianceClassifier, TARGET_CLASSES, HAS_TORCH
from services.appliances import APPLIANCES, BY_NAME
from services.command_decoder import decode_command
from services.data_loader import (
    TARGET_FS,
    load_mat_file,
    load_upload,
    preprocess,
    detect_p300,
)

app = FastAPI(title="BCI Home Appliance Prediction (EEG-TCFNet)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = ApplianceClassifier()  # loads EEG_TCFNET_CKPT if set
START_TIME = time.time()


def _preview(eeg: np.ndarray, n_channels: int = 4, n_points: int = 300) -> Dict[str, Any]:
    n_points = min(n_points, eeg.shape[1])
    idx = np.linspace(0, eeg.shape[1] - 1, n_points).astype(int)
    chs = min(n_channels, eeg.shape[0])
    return {
        "channels": [f"Ch{i + 1}" for i in range(chs)],
        "time": [round(float(t) / TARGET_FS, 4) for t in idx],
        "series": [
            {"channel": f"Ch{i + 1}", "values": [round(float(v), 4) for v in eeg[i, idx]]}
            for i in range(chs)
        ],
        "flat": [round(float(v), 4) for v in eeg[0, idx]],  # simple line-chart fallback
    }


def _predict_one(member: str, eeg_raw: np.ndarray, trig, key: str,
                 orig_fs: float, appliance_ctx: str = "") -> Dict[str, Any]:
    prep = preprocess(eeg_raw, trig, orig_fs=orig_fs, source_key=key, member_name=member)
    out = classifier.predict(prep.eeg, eeg_raw=eeg_raw, trigger=trig, orig_fs=orig_fs)
    p300, p300_score = detect_p300(prep.eeg, prep.trigger)
    ctx_name = appliance_ctx if appliance_ctx in BY_NAME else out["predicted_appliance"]
    cmd = decode_command(
        prep.eeg, prep.trigger, orig_fs,
        BY_NAME.get(ctx_name, {}).get("commands", []),
    )
    if cmd is not None:
        cmd["file"] = member
        cmd["context_appliance"] = ctx_name
    out.update(
        {
            "file": member,
            "p300_detected": bool(p300),
            "p300_score": round(float(p300_score), 4),
            "appliance_info": BY_NAME.get(out["predicted_appliance"]),
            "predicted_command": cmd,  # None when no trigger channel present
            "signal_preview": _preview(prep.eeg)["flat"],
            "signal_plot": _preview(prep.eeg),
            "meta": {
                **prep.meta,
                "orig_shape": list(prep.orig_shape),
                "orig_fs": prep.orig_fs,
                "source_key": prep.source_key,
            },
        }
    )
    return out


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "on",  # "on" while the server is running; unreachable => off
        "model": "EEG-TCFNet",
        "classes": TARGET_CLASSES,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "torch_available": HAS_TORCH,
        "checkpoint_loaded": bool(getattr(classifier, "model", None) is not None),
    }


@app.get("/api/classes")
def classes() -> Dict[str, Any]:
    return {"classes": TARGET_CLASSES}


@app.get("/api/appliances")
def appliances() -> Dict[str, Any]:
    """Catalog of what the dataset contains for each appliance application."""
    return {"appliances": APPLIANCES}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    orig_fs: float = 500.0,
    aggregate: str = "first",
    appliance: str = "",
) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith((".mat", ".zip")):
        raise HTTPException(400, "Upload must be a .mat file or .zip of .mat files.")
    try:
        blob = await file.read()
        items = load_upload(file.filename, blob)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse upload: {e}")

    try:
        if len(items) == 1 or aggregate == "first":
            name, eeg_raw, trig, key = items[0]
            result = _predict_one(name, eeg_raw, trig, key, orig_fs, appliance)
            result["files_processed"] = len(items)
            return JSONResponse(result)
        # aggregate == 'mean': average softmax over members
        probs = []
        previews = []
        first = None
        for name, eeg_raw, trig, key in items:
            r = _predict_one(name, eeg_raw, trig, key, orig_fs, appliance)
            if first is None:
                first = r
            probs.append([r["confidence_scores"][c] for c in TARGET_CLASSES])
            previews.append(name)
        mean_p = np.mean(np.array(probs), axis=0)
        mean_p = mean_p / (mean_p.sum() + 1e-12)
        cid = int(np.argmax(mean_p))
        return JSONResponse(
            {
                "predicted_appliance": TARGET_CLASSES[cid],
                "class_id": cid + 1,
                "confidence_scores": {c: round(float(p), 4) for c, p in zip(TARGET_CLASSES, mean_p)},
                "p300_detected": bool(first["p300_detected"]),
                "p300_score": first["p300_score"],
                "appliance_info": BY_NAME.get(TARGET_CLASSES[cid]),
                "predicted_command": first.get("predicted_command"),
                "signal_preview": first["signal_preview"],
                "signal_plot": first["signal_plot"],
                "meta": first["meta"],
                "files_processed": len(items),
                "files": previews,
                "model_source": "trained-ml (softmax averaged over files)",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")


@app.post("/api/predict_path")
def predict_path(path: str = Query(..., description="Server-local .mat path (dev only)")):
    """Dev helper: run inference on a dataset file already on disk."""
    try:
        eeg_raw, trig, key = load_mat_file(path)
    except Exception as e:
        raise HTTPException(400, f"Cannot load {path}: {e}")
    return JSONResponse(_predict_one(path, eeg_raw, trig, key, 500.0))


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "BCI Home Appliance Prediction (EEG-TCFNet)",
        "docs": "/docs",
        "endpoints": ["POST /api/predict", "GET /api/health", "GET /api/classes", "GET /api/appliances"],
    }
