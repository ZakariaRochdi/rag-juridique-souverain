"""
indexation/build_index.py — Construction de la base vectorielle FAISS
======================================================================
Étape 3 du pipeline : Construction de la base de connaissances.

Opérations :
  1. Chargement des chunks depuis data/processed/chunks.json
  2. Génération des embeddings avec sentence-transformers
  3. Création de l'index FAISS (IndexFlatIP pour cosine similarity)
  4. Sauvegarde de l'index + métadonnées dans data/vector_store/
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import numpy as np
from loguru import logger
from tqdm import tqdm

try:
    import faiss
except ImportError:
    raise ImportError("FAISS est requis. Installez-le avec : pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers est requis. "
        "Installez-le avec : pip install sentence-transformers"
    )


def load_chunks(processed_data_path: str) -> List[Dict[str, Any]]:
    """
    Charge les chunks depuis le fichier JSON.

    Args:
        processed_data_path: Chemin vers le répertoire data/processed/

    Returns:
        Liste de chunks avec texte et métadonnées.
    """
    chunks_path = Path(processed_data_path) / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Fichier de chunks introuvable : {chunks_path}. "
            "Exécutez d'abord : python main.py ingest"
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"📂 {len(chunks)} chunk(s) chargé(s) depuis {chunks_path}")
    return chunks


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Génère les embeddings pour tous les chunks.

    Utilise un modèle sentence-transformers multilingue
    pour encoder les textes en vecteurs denses.

    Args:
        chunks: Liste de chunks avec clé 'text'.
        config: Configuration globale (section 'embeddings').

    Returns:
        Matrice numpy (n_chunks x dimension) des embeddings.
    """
    emb_config = config.get("embeddings", {})
    model_name = emb_config.get("model_name", "intfloat/multilingual-e5-base")
    batch_size = emb_config.get("batch_size", 32)
    device = emb_config.get("device", "cpu")
    normalize = emb_config.get("normalize", True)

    logger.info(f"🤖 Chargement du modèle d'embeddings : {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Extraire les textes
    texts = [chunk["text"] for chunk in chunks]

    # Pour les modèles E5, il faut préfixer les textes
    if "e5" in model_name.lower():
        texts = [f"passage: {t}" for t in texts]
        logger.info("   → Préfixe 'passage:' ajouté pour le modèle E5")

    logger.info(
        f"📊 Génération des embeddings pour {len(texts)} chunk(s) "
        f"(batch_size={batch_size})..."
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )

    embeddings = np.array(embeddings, dtype=np.float32)
    logger.info(f"   → Shape des embeddings : {embeddings.shape}")

    return embeddings


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Crée un index FAISS à partir des embeddings.

    Utilise IndexFlatIP (Inner Product) pour la similarité cosinus
    (les vecteurs sont normalisés en amont).

    Args:
        embeddings: Matrice des embeddings (n x dim).

    Returns:
        Index FAISS prêt pour la recherche.
    """
    dimension = embeddings.shape[1]
    logger.info(f"🏗️ Création de l'index FAISS (dim={dimension})...")

    # IndexFlatIP = produit scalaire → cosine similarity si vecteurs normalisés
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    logger.info(f"   → {index.ntotal} vecteurs indexés")
    return index


def save_index(
    index: faiss.Index,
    chunks: List[Dict[str, Any]],
    vector_store_path: str,
) -> None:
    """
    Sauvegarde l'index FAISS et les métadonnées des chunks.

    Args:
        index: Index FAISS à sauvegarder.
        chunks: Liste des chunks (pour les métadonnées).
        vector_store_path: Chemin vers le répertoire de stockage.
    """
    store_path = Path(vector_store_path)
    store_path.mkdir(parents=True, exist_ok=True)

    # Sauvegarder l'index FAISS
    index_path = store_path / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    logger.info(f"💾 Index FAISS sauvegardé : {index_path}")

    # Sauvegarder les métadonnées des chunks
    metadata_path = store_path / "chunks_metadata.pkl"
    metadata = [
        {
            "id": chunk.get("id", i),
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
        }
        for i, chunk in enumerate(chunks)
    ]

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f"💾 Métadonnées sauvegardées : {metadata_path}")


def build_vector_index(config: dict) -> None:
    """
    Pipeline complet de construction de l'index vectoriel.

    1. Charge les chunks
    2. Génère les embeddings
    3. Crée l'index FAISS
    4. Sauvegarde le tout

    Args:
        config: Configuration globale du projet.
    """
    logger.info("🚀 Début de la construction de la base vectorielle")

    # 1. Charger les chunks
    chunks = load_chunks(config["paths"]["processed_data"])

    if not chunks:
        logger.warning("Aucun chunk à indexer. Abandon.")
        return

    # 2. Générer les embeddings
    embeddings = generate_embeddings(chunks, config)

    # 3. Créer l'index FAISS
    index = create_faiss_index(embeddings)

    # 4. Sauvegarder
    save_index(index, chunks, config["paths"]["vector_store"])

    logger.success(
        f"✅ Base vectorielle construite avec succès : "
        f"{index.ntotal} vecteurs de dimension {embeddings.shape[1]}"
    )


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    build_vector_index(config)
