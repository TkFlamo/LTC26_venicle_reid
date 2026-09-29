from __future__ import annotations

import io, os
from functools import lru_cache
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from .engine import VehicleSearchEngine

app = FastAPI(title='Vehicle Fingerprint ReID', version='0.5.0')


@lru_cache(maxsize=1)
def engine():
    required = ['VF_CHECKPOINT', 'VF_GALLERY_CACHE']
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError('Missing environment variables: ' + ','.join(missing))
    return VehicleSearchEngine(
        os.environ['VF_CHECKPOINT'], os.environ['VF_GALLERY_CACHE'],
        reranker=os.getenv('VF_RERANKER'), refusal=os.getenv('VF_REFUSAL'), retrieval_recipe=os.getenv('VF_RETRIEVAL_RECIPE'),
        device=os.getenv('VF_DEVICE', '0'), precision=os.getenv('VF_PRECISION', 'bf16')
    )


@app.get('/health')
def health():
    return {'status': 'ok', 'engine_loaded': engine.cache_info().currsize > 0}


@app.post('/search')
async def search(file: UploadFile = File(...), x: float | None = Form(None), y: float | None = Form(None),
                 w: float | None = Form(None), h: float | None = Form(None), topk: int = Form(10)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    except Exception as e:
        raise HTTPException(400, f'Invalid image: {e}')
    vals = [x, y, w, h]
    bbox = None
    if any(v is not None for v in vals):
        if not all(v is not None for v in vals):
            raise HTTPException(400, 'Provide all x,y,w,h or none')
        bbox = (x, y, w, h)
    return engine().search(image, bbox=bbox, topk=max(1, min(50, topk)))
