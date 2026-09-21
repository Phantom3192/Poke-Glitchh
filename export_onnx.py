"""
export_onnx.py - Export models/pokemon_classifier.pt to ONNX so main.py can run it with
ONNX Runtime (faster on CPU than PyTorch eager, especially EfficientNet's depthwise convs).

    python export_onnx.py                       # export + verify against PyTorch
    python export_onnx.py --images "Extra pokemons.zip"   # verify on real images (dir or zip)
    python export_onnx.py --bench               # also time PyTorch vs ONNX Runtime

main.py also calls export_to_onnx() by itself on startup when the .onnx is missing or was
exported from a different .pt (AUTO_EXPORT_ONNX=true, the default), so running this by hand
is optional. If verification fails the .onnx is deleted and the bot keeps using PyTorch.
"""

import io
import os
import sys
import glob
import time
import zipfile
import argparse
from typing import List, Optional

import numpy as np
from PIL import Image

from onnx_backend import OnnxExtractor, write_meta, INPUT_SIZE

COS_THRESHOLD = 0.999


def export_to_onnx(extractor, out_path: str, source_path: Optional[str] = None, opset: int = 17,
                   verify_images: Optional[List[Image.Image]] = None) -> dict:
    """
    Exports a loaded PokemonFeatureExtractor (train_model.py) to `out_path`, verifies the ONNX
    embeddings match PyTorch's, and writes a sidecar recording which .pt it came from.
    Raises RuntimeError (and removes the file) if verification fails.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Wrapped(nn.Module):
        """normalize -> backbone -> projection -> L2 normalize, on a [0,1] (N,3,224,224) tensor."""
        def __init__(self, ex):
            super().__init__()
            self.backbone, self.projection = ex.backbone, ex.projection
            self.register_buffer("mean", torch.tensor(list(ex.normalize.mean), dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(list(ex.normalize.std), dtype=torch.float32).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            return F.normalize(self.projection(self.backbone(x)), p=2, dim=1)

    model = Wrapped(extractor).eval()
    dummy = torch.rand(2, 3, INPUT_SIZE, INPUT_SIZE)
    kwargs = dict(input_names=["image"], output_names=["embedding"],
                  dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=opset)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + ".tmp"
    with torch.no_grad():
        try:
            torch.onnx.export(model, (dummy,), tmp, dynamo=False, **kwargs)
        except TypeError:  # older torch has no `dynamo` argument
            torch.onnx.export(model, (dummy,), tmp, **kwargs)
    os.replace(tmp, out_path)

    try:
        stats = verify(extractor, out_path, verify_images)
    except Exception:
        _remove(out_path)
        raise
    if stats["min_cos"] < COS_THRESHOLD:
        _remove(out_path)
        raise RuntimeError(f"ONNX embeddings differ from PyTorch (min cosine {stats['min_cos']:.6f} < {COS_THRESHOLD})")

    if source_path and os.path.exists(source_path):
        write_meta(out_path, source_path, {"opset": opset, "torch": torch.__version__})
    return stats


def _remove(path: str):
    for p in (path, path + ".json", path + ".tmp"):
        try:
            os.remove(p)
        except OSError:
            pass


def synthetic_images(n: int = 12) -> List[Image.Image]:
    rng = np.random.default_rng(0)
    imgs = []
    for i in range(n):
        w, h = int(rng.integers(120, 420)), int(rng.integers(120, 420))
        base = rng.integers(0, 255, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
        img = Image.fromarray(base).resize((w, h), Image.Resampling.BICUBIC)   # smooth-ish, like art
        imgs.append(img.convert("RGBA").convert("RGB") if i % 3 == 0 else img)
    return imgs


def verify(extractor, onnx_path: str, images: Optional[List[Image.Image]] = None, threads: int = 1) -> dict:
    """Compare PyTorch vs ONNX Runtime embeddings (single + batched) on the same images."""
    images = [im.convert("RGB") for im in (images or synthetic_images())]
    ort_ex = OnnxExtractor(onnx_path, threads=threads)
    ref = np.stack([extractor.extract(im) for im in images]).astype(np.float32)
    single = np.stack([ort_ex.extract(im) for im in images])
    batched = ort_ex.extract_batch(images)
    cos = (ref * single).sum(1)                       # both L2-normalised
    return {
        "n": len(images),
        "min_cos": float(cos.min()),
        "mean_cos": float(cos.mean()),
        "max_abs_diff": float(np.abs(ref - single).max()),
        "batch_vs_single_max_abs_diff": float(np.abs(batched - single).max()),
    }


def load_images(source: str, limit: int = 24) -> List[Image.Image]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    imgs: List[Image.Image] = []
    if source.lower().endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(exts)]
            step = max(1, len(names) // limit)
            for n in names[::step][:limit]:
                try:
                    imgs.append(Image.open(io.BytesIO(zf.read(n))).convert("RGB"))
                except Exception:
                    continue
    else:
        paths = sorted(p for e in exts for p in glob.glob(os.path.join(source, "**", f"*{e}"), recursive=True))
        step = max(1, len(paths) // limit)
        for p in paths[::step][:limit]:
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                continue
    return imgs


def bench(extractor, onnx_path: str, images: List[Image.Image], threads: int = 1, batch: int = 8):
    import torch
    torch.set_num_threads(threads)
    ort_ex = OnnxExtractor(onnx_path, threads=threads)
    imgs = (images * (batch * 4 // max(1, len(images)) + 1))[: batch * 4]

    def timeit(fn):
        fn()  # warm-up
        t = time.perf_counter()
        fn()
        return (time.perf_counter() - t) * 1000 / len(imgs)

    res = {
        "torch (1 img)": timeit(lambda: [extractor.extract(i) for i in imgs]),
        "onnx  (1 img)": timeit(lambda: [ort_ex.extract(i) for i in imgs]),
        f"onnx  (batch {batch})": timeit(lambda: [ort_ex.extract_batch(imgs[k:k + batch]) for k in range(0, len(imgs), batch)]),
    }
    print(f"\nms per image ({threads} thread(s), preprocessing included):")
    for k, v in res.items():
        print(f"  {k:<16s} {v:7.1f} ms   ({1000 / v:5.1f} img/s)")


def main():
    ap = argparse.ArgumentParser(description="Export the Pokemon feature extractor to ONNX")
    ap.add_argument("--out", default=None, help="default: ONNX_MODEL_PATH env, else next to the .pt")
    ap.add_argument("--model", default=None, help="path to the .pt (default: MODEL_OUTPUT from train_model.py)")
    ap.add_argument("--images", default=None, help="dir or .zip of real images to verify on")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    from predict import load_extractor
    from train_model import MODEL_OUTPUT
    model_path = args.model or MODEL_OUTPUT
    args.out = args.out or os.getenv("ONNX_MODEL_PATH") or os.path.splitext(model_path)[0] + ".onnx"

    print(f"📦 Loading PyTorch model from {model_path} ...")
    extractor = load_extractor(model_path)

    src = args.images or ("Extra pokemons.zip" if os.path.exists("Extra pokemons.zip") else None)
    real = load_images(src) if src else []
    print(f"🧪 Verifying on {len(real)} real image(s) + synthetic ones" if real else "🧪 Verifying on synthetic images only")

    print(f"🔧 Exporting to {args.out} (opset {args.opset}) ...")
    stats = export_to_onnx(extractor, args.out, model_path, args.opset, (real or []) + synthetic_images())
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"✅ Exported ({size_mb:.1f} MB). Verified on {stats['n']} images:")
    print(f"   min cosine vs PyTorch : {stats['min_cos']:.7f}")
    print(f"   max |diff|            : {stats['max_abs_diff']:.2e}")
    print(f"   batch vs single |diff|: {stats['batch_vs_single_max_abs_diff']:.2e}")

    if args.bench:
        bench(extractor, args.out, real or synthetic_images(), args.threads)
    print("\nCommit models/pokemon_classifier.onnx and models/pokemon_classifier.onnx.json with the .pt "
          "(or leave them out: main.py re-exports on startup).")


if __name__ == "__main__":
    sys.exit(main())
