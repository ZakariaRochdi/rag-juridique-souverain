"""
tests/test_retrieval.py — Tests unitaires du module de recherche sémantique
============================================================================
Teste les fonctionnalités du retriever :
  - Encodage des questions
  - Recherche sémantique
  - Filtrage par seuil de similarité
  - Gestion des cas limites
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    """Configuration de test."""
    return {
        "paths": {
            "vector_store": "data/vector_store",
            "processed_data": "data/processed",
        },
        "embeddings": {
            "model_name": "intfloat/multilingual-e5-base",
            "dimension": 768,
            "batch_size": 32,
            "normalize": True,
            "device": "cpu",
        },
        "retrieval": {
            "top_k": 5,
            "similarity_threshold": 0.3,
            "search_type": "similarity",
        },
    }


@pytest.fixture
def sample_chunks():
    """Chunks de test simulant des articles juridiques."""
    return [
        {
            "id": 0,
            "text": "Article 152 : La salariée a droit à un congé de maternité de quatorze semaines.",
            "metadata": {
                "source": "code-travail-maroc.pdf",
                "article": "Article 152",
                "tokens_estimate": 18,
            },
        },
        {
            "id": 1,
            "text": "Article 184 : La durée normale de travail est fixée à 2496 heures par année.",
            "metadata": {
                "source": "code-travail-maroc.pdf",
                "article": "Article 184",
                "tokens_estimate": 17,
            },
        },
        {
            "id": 2,
            "text": "Article 231 : Le salarié a droit à un congé annuel payé d'un jour et demi par mois.",
            "metadata": {
                "source": "code-travail-maroc.pdf",
                "article": "Article 231",
                "tokens_estimate": 20,
            },
        },
    ]


@pytest.fixture
def mock_faiss_index():
    """Index FAISS simulé."""
    index = MagicMock()
    index.ntotal = 3
    # Simuler une recherche retournant 3 résultats avec des scores
    index.search.return_value = (
        np.array([[0.92, 0.65, 0.40]], dtype=np.float32),
        np.array([[0, 2, 1]], dtype=np.int64),
    )
    return index


# ─── Tests de la recherche sémantique ──────────────────────────────

class TestSemanticRetriever:
    """Tests pour la classe SemanticRetriever."""

    @patch("retrieval.retriever.faiss")
    @patch("retrieval.retriever.SentenceTransformer")
    def test_search_returns_results(
        self, mock_st, mock_faiss, mock_config, sample_chunks, mock_faiss_index
    ):
        """Vérifie que la recherche retourne des résultats pertinents."""
        # Setup mocks
        mock_st.return_value.encode.return_value = np.random.rand(1, 768).astype(
            np.float32
        )
        mock_faiss.read_index.return_value = mock_faiss_index

        from retrieval.retriever import SemanticRetriever

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("pickle.load", return_value=sample_chunks):
                    retriever = SemanticRetriever(mock_config)

        results = retriever.search("congé de maternité")

        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all("metadata" in r for r in results)

    @patch("retrieval.retriever.faiss")
    @patch("retrieval.retriever.SentenceTransformer")
    def test_search_respects_top_k(
        self, mock_st, mock_faiss, mock_config, sample_chunks, mock_faiss_index
    ):
        """Vérifie que le nombre de résultats respecte top_k."""
        mock_st.return_value.encode.return_value = np.random.rand(1, 768).astype(
            np.float32
        )
        mock_faiss.read_index.return_value = mock_faiss_index

        from retrieval.retriever import SemanticRetriever

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("pickle.load", return_value=sample_chunks):
                    retriever = SemanticRetriever(mock_config)

        results = retriever.search("congé de maternité", top_k=2)

        # top_k=2 mais le threshold pourrait filtrer certains résultats
        assert len(results) <= 2

    @patch("retrieval.retriever.faiss")
    @patch("retrieval.retriever.SentenceTransformer")
    def test_search_filters_by_threshold(
        self, mock_st, mock_faiss, mock_config, sample_chunks
    ):
        """Vérifie que le seuil de similarité filtre les résultats."""
        mock_st.return_value.encode.return_value = np.random.rand(1, 768).astype(
            np.float32
        )

        # Index avec des scores bas
        low_score_index = MagicMock()
        low_score_index.ntotal = 3
        low_score_index.search.return_value = (
            np.array([[0.15, 0.10, 0.05]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )
        mock_faiss.read_index.return_value = low_score_index

        from retrieval.retriever import SemanticRetriever

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("pickle.load", return_value=sample_chunks):
                    retriever = SemanticRetriever(mock_config)

        # Threshold à 0.3, tous les scores sont en dessous
        results = retriever.search("question test", threshold=0.3)
        assert len(results) == 0

    @patch("retrieval.retriever.faiss")
    @patch("retrieval.retriever.SentenceTransformer")
    def test_search_results_sorted_by_score(
        self, mock_st, mock_faiss, mock_config, sample_chunks, mock_faiss_index
    ):
        """Vérifie que les résultats sont triés par score décroissant."""
        mock_st.return_value.encode.return_value = np.random.rand(1, 768).astype(
            np.float32
        )
        mock_faiss.read_index.return_value = mock_faiss_index

        from retrieval.retriever import SemanticRetriever

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("pickle.load", return_value=sample_chunks):
                    retriever = SemanticRetriever(mock_config)

        results = retriever.search("congé de maternité", threshold=0.0)

        if len(results) > 1:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    @patch("retrieval.retriever.faiss")
    @patch("retrieval.retriever.SentenceTransformer")
    def test_search_with_context(
        self, mock_st, mock_faiss, mock_config, sample_chunks, mock_faiss_index
    ):
        """Vérifie la construction du contexte agrégé."""
        mock_st.return_value.encode.return_value = np.random.rand(1, 768).astype(
            np.float32
        )
        mock_faiss.read_index.return_value = mock_faiss_index

        from retrieval.retriever import SemanticRetriever

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("pickle.load", return_value=sample_chunks):
                    retriever = SemanticRetriever(mock_config)

        result = retriever.search_with_context("congé de maternité")

        assert "context" in result
        assert "sources" in result
        assert "results" in result
        assert "has_relevant_results" in result

    @patch("retrieval.retriever.faiss")
    @patch("retrieval.retriever.SentenceTransformer")
    def test_e5_prefix_added(
        self, mock_st, mock_faiss, mock_config, sample_chunks, mock_faiss_index
    ):
        """Vérifie que le préfixe E5 est ajouté pour les modèles E5."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        mock_st.return_value = mock_model
        mock_faiss.read_index.return_value = mock_faiss_index

        from retrieval.retriever import SemanticRetriever

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("pickle.load", return_value=sample_chunks):
                    retriever = SemanticRetriever(mock_config)

        retriever.search("test")

        # Vérifier que le modèle a été appelé avec le préfixe "query: "
        call_args = mock_model.encode.call_args
        encoded_text = call_args[0][0][0]
        assert encoded_text.startswith("query: ")


class TestRetrieverEdgeCases:
    """Tests des cas limites pour le retriever."""

    @patch("retrieval.retriever.faiss")
    @patch("retrieval.retriever.SentenceTransformer")
    def test_empty_query(
        self, mock_st, mock_faiss, mock_config, sample_chunks, mock_faiss_index
    ):
        """Vérifie le comportement avec une question vide."""
        mock_st.return_value.encode.return_value = np.random.rand(1, 768).astype(
            np.float32
        )
        mock_faiss.read_index.return_value = mock_faiss_index

        from retrieval.retriever import SemanticRetriever

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("pickle.load", return_value=sample_chunks):
                    retriever = SemanticRetriever(mock_config)

        results = retriever.search("")
        # Le système ne devrait pas planter, même avec une question vide
        assert isinstance(results, list)

    @patch("retrieval.retriever.faiss")
    @patch("retrieval.retriever.SentenceTransformer")
    def test_no_results_context(
        self, mock_st, mock_faiss, mock_config, sample_chunks
    ):
        """Vérifie le contexte quand aucun résultat n'est pertinent."""
        mock_st.return_value.encode.return_value = np.random.rand(1, 768).astype(
            np.float32
        )

        empty_index = MagicMock()
        empty_index.ntotal = 0
        empty_index.search.return_value = (
            np.array([[]], dtype=np.float32).reshape(1, 0),
            np.array([[]], dtype=np.int64).reshape(1, 0),
        )
        mock_faiss.read_index.return_value = empty_index

        from retrieval.retriever import SemanticRetriever

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", MagicMock()):
                with patch("pickle.load", return_value=sample_chunks):
                    retriever = SemanticRetriever(mock_config)

        result = retriever.search_with_context("question sans réponse")
        assert result["has_relevant_results"] is False
        assert result["context"] == ""
