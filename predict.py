"""
predict.py - Use the trained Pokemon classifier to identify a species from an image.

How it works:
    1. Loads the trained EfficientNet-B0 feature extractor from models/pokemon_classifier.pt
    2. Embeds the query image into the same 256-dim vector space used at training time
    3. Loads every stored per-image embedding from the DB (pokemon_features table)
    4. Finds the closest matches by cosine similarity and returns a majority vote
       over the top-k neighbors (more robust than a single nearest neighbor)

Usage:
    python predict.py path/to/image.jpg
    python predict.py path/to/image.jpg --top 10 --show-all
    python predict.py path/to/dir_of_images/

Environment (same as train_model.py):
    TURSO_URL, TURSO_AUTH_TOKEN   -> use Turso if set
    DB_PATH                       -> else fall back to local SQLite file (default pokemon.db)
    MODEL_OUTPUT                  -> path to the saved model (default models/pokemon_classifier.pt)
"""

import os
import sys
import json
import glob
import argparse
from collections import Counter

import numpy as np
import torch
from PIL import Image

# Reuse the exact classes/constants train_model.py uses, so preprocessing and
# DB access stay identical between training and inference.
from train_model import PokemonFeatureExtractor, Database, MODEL_OUTPUT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_extractor(model_path: str) -> PokemonFeatureExtractor:
    if not os.path.exists(model_path):
        sys.exit(f"❌ Model file not found: {model_path}\n"
                  f"   Run train_model.py first, or pass --model to point at the right file.")
    extractor = PokemonFeatureExtractor(embedding_dim=256)
    state_dict = torch.load(model_path, map_location="cpu")
    extractor.load_state_dict(state_dict)
    extractor.eval()
    return extractor


def load_feature_bank(db: Database):
    """
    Pulls every stored (species, vector) pair once into memory as a single
    NumPy matrix, so a lookup is one matrix multiply instead of N per-row
    queries. Cache this across multiple predictions in the same run.
    """
    cursor = db._conn.cursor()
    cursor.execute("SELECT species, variant_name, feature_vector FROM pokemon_features")
    rows = cursor.fetchall()
    if not rows:
        sys.exit("❌ No features found in the database. Did training finish and write to the DB?")

    species_list = []
    vectors = []
    for row in rows:
        # sqlite3.Row supports both index and key access; libsql rows are
        # plain tuples, so index access works for either backend.
        species = row[0]
        feature_vector = row[2]
        species_list.append(species)
        vectors.append(json.loads(feature_vector))

    matrix = np.asarray(vectors, dtype=np.float32)  # (N, 256), already L2-normalized at write time
    return species_list, matrix


def predict_species(query_vec: np.ndarray, species_list, matrix: np.ndarray, top_k: int = 5):
    """
    Cosine similarity == plain dot product here since every stored vector
    (and the query vector) is already L2-normalized.
    """
    sims = matrix @ query_vec  # (N,)
    order = np.argsort(-sims)[:top_k]

    neighbors = [(species_list[i], float(sims[i])) for i in order]

    # Majority vote across the top-k neighbors, tie-broken by best single similarity
    votes = Counter(sp for sp, _ in neighbors)
    winner, _ = max(
        votes.items(),
        key=lambda kv: (kv[1], max(s for sp, s in neighbors if sp == kv[0]))
    )
    best_score = max(s for sp, s in neighbors if sp == winner)
    return winner, best_score, neighbors


def iter_image_paths(path: str):
    if os.path.isdir(path):
        for ext in IMAGE_EXTS:
            yield from glob.glob(os.path.join(path, f"*{ext}"))
    else:
        yield path


def main():
    parser = argparse.ArgumentParser(description="Identify a Pokemon species from an image.")
    parser.add_argument("image_path", help="Path to an image file, or a directory of images")
    parser.add_argument("--model", default=MODEL_OUTPUT, help="Path to the trained model file")
    parser.add_argument("--top", type=int, default=5, help="Number of nearest neighbors to vote over")
    parser.add_argument("--show-all", action="store_true", help="Print all top-k neighbors, not just the winner")
    args = parser.parse_args()

    print(f"📦 Loading model from {args.model} ...")
    extractor = load_extractor(args.model)

    print("🗄️  Connecting to database and loading feature bank ...")
    db = Database()
    species_list, matrix = load_feature_bank(db)
    print(f"   Loaded {matrix.shape[0]} feature vectors across {len(set(species_list))} species\n")

    for img_path in iter_image_paths(args.image_path):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️  Skipping {img_path}: {e}")
            continue

        query_vec = extractor.extract(img)
        winner, score, neighbors = predict_species(query_vec, species_list, matrix, top_k=args.top)

        print(f"🔍 {os.path.basename(img_path)}")
        print(f"   → Predicted: {winner}  (similarity {score:.3f})")
        if args.show_all:
            for sp, sim in neighbors:
                print(f"      {sp:<20s} {sim:.3f}")
        print()

    db.close()


if __name__ == "__main__":
    main()
