"""
retrieval/retriever.py — Recherche sémantique des passages pertinents
======================================================================
Étape 4 du pipeline : Recherche des informations pertinentes.

Le retriever :
  1. Encode la question utilisateur en embedding
  2. Recherche les top-K chunks les plus similaires dans FAISS
  3. Filtre par seuil de similarité
  4. Retourne les passages pertinents avec scores et métadonnées
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import numpy as np
from loguru import logger

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None



class SemanticRetriever:
    """
    Moteur de recherche sémantique sur la base vectorielle FAISS.

    Encode les questions en embeddings et recherche les chunks les
    plus similaires par produit scalaire (cosine similarity).
    """

    def __init__(self, config: dict):
        """
        Initialise le retriever en chargeant l'index et le modèle.

        Args:
            config: Configuration globale du projet.
        """
        self.config = config
        self.retrieval_config = config.get("retrieval", {})
        self.emb_config = config.get("embeddings", {})

        self.top_k = self.retrieval_config.get("top_k", 5)
        self.similarity_threshold = self.retrieval_config.get(
            "similarity_threshold", 0.3
        )

        # Charger le modèle d'embeddings
        model_name = self.emb_config.get(
            "model_name", "intfloat/multilingual-e5-base"
        )
        device = self.emb_config.get("device", "cpu")

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers est requis. Installez-le avec : pip install sentence-transformers"
            )
        if faiss is None:
            raise ImportError("FAISS est requis. Installez-le avec : pip install faiss-cpu")

        logger.info(f"🤖 Chargement du modèle d'embeddings : {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.is_e5 = "e5" in model_name.lower()


        # Charger l'index FAISS
        self._load_index()

    def _load_index(self) -> None:
        """Charge l'index FAISS et les métadonnées depuis le disque."""
        vector_store_path = Path(self.config["paths"]["vector_store"])

        # Index FAISS
        index_path = vector_store_path / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index FAISS introuvable : {index_path}. "
                "Exécutez d'abord : python main.py index"
            )

        self.index = faiss.read_index(str(index_path))
        logger.info(f"📂 Index FAISS chargé : {self.index.ntotal} vecteurs")

        # Métadonnées
        metadata_path = vector_store_path / "chunks_metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Métadonnées introuvables : {metadata_path}"
            )

        with open(metadata_path, "rb") as f:
            self.chunks_metadata = pickle.load(f)

        logger.info(f"   → {len(self.chunks_metadata)} chunk(s) de métadonnées chargé(s)")

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode une question en vecteur d'embedding.

        Pour les modèles E5, ajoute le préfixe 'query:'.

        Args:
            query: Question en langage naturel.

        Returns:
            Vecteur embedding normalisé (1 x dim).
        """
        if self.is_e5:
            query = f"query: {query}"

        embedding = self.model.encode(
            [query],
            normalize_embeddings=self.emb_config.get("normalize", True),
        )

        return np.array(embedding, dtype=np.float32)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recherche les passages les plus pertinents pour une question.

        Args:
            query: Question en langage naturel.
            top_k: Nombre de résultats à retourner (défaut: config).
            threshold: Seuil minimal de similarité (défaut: config).

        Returns:
            Liste de résultats triés par score décroissant, chacun contenant :
              - text: texte du chunk
              - score: score de similarité cosinus
              - metadata: métadonnées du chunk (source, article, etc.)
        """
        k = top_k or self.top_k
        min_score = threshold or self.similarity_threshold

        logger.info(f"🔍 Recherche pour : '{query[:80]}...' (top_k={k})")

        # Encoder la question
        query_embedding = self._encode_query(query)

        # Recherche dans FAISS
        scores, indices = self.index.search(query_embedding, k)

        # Construire les résultats
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS retourne -1 si pas assez de résultats
                continue

            if score < min_score:
                logger.debug(
                    f"   Chunk {idx} ignoré (score={score:.4f} < seuil={min_score})"
                )
                continue

            chunk_data = self.chunks_metadata[idx]
            results.append({
                "text": chunk_data["text"],
                "score": float(score),
                "metadata": chunk_data.get("metadata", {}),
                "chunk_id": chunk_data.get("id", idx),
            })

        logger.info(
            f"   → {len(results)} résultat(s) au-dessus du seuil ({min_score})"
        )

        # Log des scores
        for r in results:
            article = r["metadata"].get("article", "?")
            logger.debug(f"   📎 [{article}] score={r['score']:.4f}")

        return results[:k]

    def search_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Recherche et construit un contexte agrégé pour la génération.

        Args:
            query: Question en langage naturel.
            top_k: Nombre de résultats.

        Returns:
            Dictionnaire contenant :
              - context: texte de contexte agrégé
              - sources: liste des sources uniques
              - results: résultats bruts
              - has_relevant_results: booléen
        """
        results = self.search(query, top_k)

        if not results:
            return {
                "context": "",
                "sources": [],
                "results": [],
                "has_relevant_results": False,
            }

        # Construire le contexte agrégé
        context_parts = []
        sources = set()

        for i, r in enumerate(results):
            source = r["metadata"].get("source", "Source inconnue")
            article = r["metadata"].get("article", "")
            source_ref = f"{source}, {article}" if article else source

            context_parts.append(
                f"[Extrait {i + 1}] ({source_ref}) :\n{r['text']}"
            )
            sources.add(source_ref)

        context = "\n\n---\n\n".join(context_parts)

        return {
            "context": context,
            "sources": sorted(sources),
            "results": results,
            "has_relevant_results": True,
        }


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    retriever = SemanticRetriever(config)

    # Test
    query = "Quelle est la durée du congé de maternité ?"
    results = retriever.search(query)

    print(f"\n🔍 Question : {query}")
    print(f"📊 {len(results)} résultat(s) trouvé(s)\n")

    for r in results:
        print(f"   📎 [{r['metadata'].get('article', '?')}] score={r['score']:.4f}")
        print(f"      {r['text'][:150]}...\n")
