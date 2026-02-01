from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from mmbiometric.inference.predictor import Predictor

app = FastAPI(title="mmbiometric inference")

PREDICTOR: Predictor | None = None


def _load_predictor() -> Predictor:
    model_dir = Path(os.environ.get("MODEL_DIR", "/model"))
    ckpt = model_dir / "best.pt"
    labels = model_dir / "labels.json"

    if not ckpt.exists() or not labels.exists():
        raise RuntimeError(f"Missing model files in {model_dir}")

    idx_to_label_raw: dict[str, Any] = json.loads(labels.read_text(encoding="utf-8"))
    idx_to_label = {int(k): str(v) for k, v in idx_to_label_raw.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Predictor.load(
        ckpt_path=ckpt,
        backbone=os.environ.get("BACKBONE", "resnet18"),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "256")),
        dropout=float(os.environ.get("DROPOUT", "0.1")),
        idx_to_label=idx_to_label,
        image_size=int(os.environ.get("IMAGE_SIZE", "224")),
        device=device,
    )


@app.on_event("startup")
def startup() -> None:
    global PREDICTOR
    PREDICTOR = _load_predictor()


@app.post("/predict")
async def predict(
    iris: UploadFile = File(...),
    fingerprint: UploadFile = File(...),
) -> dict[str, str]:
    if PREDICTOR is None:
        raise HTTPException(status_code=500, detail="Predictor not initialized")

    iris_img = Image.open(io.BytesIO(await iris.read()))
    fp_img = Image.open(io.BytesIO(await fingerprint.read()))
    pred = PREDICTOR.predict_one(iris_img, fp_img)
    return {"prediction": pred}
