"""
onnx_backend.py - ONNX Runtime drop-in for PokemonFeatureExtractor (inference only).

The exported graph is: (N, 3, 224, 224) float32 in [0, 1]  ->  normalize -> EfficientNet-B0
-> projection -> L2 normalize -> (N, 256). It is the SAME network as models/pokemon_classifier.pt,
so embeddings match the feature bank stored in the DB (within ~1e-6).

Preprocessing is done here with PIL + numpy and mirrors train_model.py exactly
(Resize((224, 224)) bilinear -> ToTensor), so torch/torchvision aren't needed at inference.

Exposes the two methods main.py uses:
    extract(img)          -> (256,) float32 vector   (zeros on failure)
    extract_batch(imgs)   -> (N, 256) float32        (zeros rows for None / unreadable images,
                                                      row i always belongs to imgs[i])
"""

import os
import json
import hashlib
import logging
from typing import List, Optional

import numpy as np
from PIL import Image

log = logging.getLogger("onnx_backend")

INPUT_SIZE = 224
_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta_path(onnx_path: str) -> str:
    return onnx_path + ".json"


def write_meta(onnx_path: str, source_path: str, extra: Optional[dict] = None):
    meta = {"source_sha1": file_sha1(source_path), "source_size": os.path.getsize(source_path)}
    meta.update(extra or {})
    with open(meta_path(onnx_path), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def onnx_matches_source(onnx_path: str, source_path: str) -> Optional[bool]:
    """True/False if the .onnx was exported from this exact .pt; None if unknown (no sidecar)."""
    try:
        with open(meta_path(onnx_path), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("source_sha1") == file_sha1(source_path)
    except (OSError, ValueError):
        return None


class OnnxExtractor:
    def __init__(self, onnx_path: str, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 1:
            # don't burn CPU spinning between requests inside a shared container
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        out_dim = self.session.get_outputs()[0].shape[-1]
        self.dim = int(out_dim) if isinstance(out_dim, int) else 256
        # first run allocates buffers; do it now instead of on the first live spawn
        self.session.run(None, {self.input_name: np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)})

    @staticmethod
    def _prep(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), _BILINEAR)
        return np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0

    def extract_batch(self, images: List[Optional[Image.Image]]) -> np.ndarray:
        out = np.zeros((len(images), self.dim), dtype=np.float32)
        idx, arrs = [], []
        for i, img in enumerate(images):
            try:
                arrs.append(self._prep(img))
                idx.append(i)
            except Exception:
                continue  # None / unreadable image -> stays a zero row
        if not arrs:
            return out
        batch = np.ascontiguousarray(np.stack(arrs))
        try:
            emb = self.session.run(None, {self.input_name: batch})[0]
        except Exception as e:
            log.warning(f"ONNX batch inference failed: {type(e).__name__}: {e}")
            return out
        out[idx] = emb
        return out

    def extract(self, img: Optional[Image.Image]) -> np.ndarray:
        return self.extract_batch([img])[0]
