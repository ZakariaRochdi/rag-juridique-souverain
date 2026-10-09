"""
tests/test_evaluation.py — Tests unitaires du module d'évaluation
==================================================================
Teste les métriques d'évaluation :
  - Precision@K, Recall@K, MRR
  - ROUGE-L
  - Faithfulness (fidélité)
  - Abstention accuracy
"""

import sys
from pathlib import Path

import pytest

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    rouge_l_score,
    compute_faithfulness,
    abstention_accuracy,
)


# ═══════════════════════════════════════════════════════════════════
# Tests des métriques de Retrieval
# ═══════════════════════════════════════════════════════════════════

class TestPrecisionAtK:
    """Tests pour Precision@K."""

    def test_perfect_precision(self):
        """Precision@1 = 1.0 quand le premier résultat est pertinent."""
        retrieved = ["Code du travail, Article 152"]
        relevant = ["Code du travail, Article 152"]
        assert precision_at_k(retrieved, relevant, 1) == 1.0

    def test_zero_precision(self):
        """Precision@1 = 0.0 quand aucun résultat n'est pertinent."""
        retrieved = ["Code du travail, Article 999"]
        relevant = ["Code du travail, Article 152"]
        assert precision_at_k(retrieved, relevant, 1) == 0.0

    def test_partial_precision(self):
        """Precision@2 = 0.5 quand 1/2 résultats est pertinent."""
        retrieved = [
            "Code du travail, Article 152",
            "Code du travail, Article 999",
        ]
        relevant = ["Code du travail, Article 152"]
        assert precision_at_k(retrieved, relevant, 2) == 0.5

    def test_precision_k_zero(self):
        """Precision@0 = 0.0."""
        assert precision_at_k(["A"], ["A"], 0) == 0.0

    def test_precision_empty_retrieved(self):
        """Precision avec liste de résultats vide."""
        assert precision_at_k([], ["A"], 1) == 0.0

    def test_precision_empty_relevant(self):
        """Precision avec aucune source pertinente."""
        assert precision_at_k(["A", "B"], [], 2) == 0.0

    def test_precision_case_insensitive(self):
        """Vérifie la normalisation des sources (insensible à la casse)."""
        retrieved = ["CODE DU TRAVAIL, article 152"]
        relevant = ["code du travail, Article 152"]
        assert precision_at_k(retrieved, relevant, 1) == 1.0


class TestRecallAtK:
    """Tests pour Recall@K."""

    def test_perfect_recall(self):
        """Recall@1 = 1.0 quand toutes les sources pertinentes sont trouvées."""
        retrieved = ["Code du travail, Article 152"]
        relevant = ["Code du travail, Article 152"]
        assert recall_at_k(retrieved, relevant, 1) == 1.0

    def test_zero_recall(self):
        """Recall@1 = 0.0 quand aucune source pertinente n'est trouvée."""
        retrieved = ["Code du travail, Article 999"]
        relevant = ["Code du travail, Article 152"]
        assert recall_at_k(retrieved, relevant, 1) == 0.0

    def test_partial_recall(self):
        """Recall quand seule une partie des sources est retrouvée."""
        retrieved = ["Code du travail, Article 152"]
        relevant = [
            "Code du travail, Article 152",
            "Code du travail, Article 153",
        ]
        assert recall_at_k(retrieved, relevant, 1) == 0.5

    def test_recall_empty_relevant(self):
        """Recall avec aucune source pertinente (division par zéro)."""
        assert recall_at_k(["A"], [], 1) == 0.0


class TestMRR:
    """Tests pour Mean Reciprocal Rank."""

    def test_first_position(self):
        """MRR = 1.0 quand le premier résultat est pertinent."""
        retrieved = ["A", "B", "C"]
        relevant = ["A"]
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_position(self):
        """MRR = 0.5 quand le premier résultat pertinent est en 2ème position."""
        retrieved = ["X", "A", "B"]
        relevant = ["A"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_third_position(self):
        """MRR = 1/3 quand le premier résultat pertinent est en 3ème position."""
        retrieved = ["X", "Y", "A"]
        relevant = ["A"]
        assert abs(mean_reciprocal_rank(retrieved, relevant) - 1 / 3) < 1e-6

    def test_no_relevant(self):
        """MRR = 0.0 quand aucun résultat n'est pertinent."""
        retrieved = ["X", "Y", "Z"]
        relevant = ["A"]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_empty_retrieved(self):
        """MRR = 0.0 avec une liste vide de résultats."""
        assert mean_reciprocal_rank([], ["A"]) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Tests des métriques de Génération
# ═══════════════════════════════════════════════════════════════════

class TestROUGEL:
    """Tests pour ROUGE-L."""

    def test_identical_texts(self):
        """ROUGE-L = 1.0 pour des textes identiques."""
        text = "La salariée a droit à un congé de maternité."
        assert rouge_l_score(text, text) == 1.0

    def test_completely_different(self):
        """ROUGE-L proche de 0 pour des textes très différents."""
        gen = "Le ciel est bleu et le soleil brille."
        ref = "Python est un langage de programmation populaire."
        score = rouge_l_score(gen, ref)
        assert score < 0.2

    def test_partial_overlap(self):
        """ROUGE-L > 0 pour des textes avec un chevauchement partiel."""
        gen = "La salariée a droit à un congé de maternité de 14 semaines."
        ref = "La salariée bénéficie d'un congé de maternité de quatorze semaines."
        score = rouge_l_score(gen, ref)
        assert 0.0 < score < 1.0

    def test_empty_generated(self):
        """ROUGE-L = 0.0 avec une réponse générée vide."""
        assert rouge_l_score("", "Texte de référence") == 0.0

    def test_empty_reference(self):
        """ROUGE-L = 0.0 avec une référence vide."""
        assert rouge_l_score("Texte généré", "") == 0.0

    def test_both_empty(self):
        """ROUGE-L = 0.0 avec les deux textes vides."""
        assert rouge_l_score("", "") == 0.0


class TestFaithfulness:
    """Tests pour la fidélité aux sources."""

    def test_fully_faithful(self):
        """Score élevé quand la réponse utilise uniquement le contexte."""
        context = "Article 152 : La salariée bénéficie d'un congé de maternité de quatorze semaines."
        generated = "La salariée bénéficie d'un congé de maternité de quatorze semaines selon l'article 152."
        score = compute_faithfulness(generated, context)
        assert score > 0.7

    def test_unfaithful(self):
        """Score bas quand la réponse n'est pas dans le contexte."""
        context = "Article 152 : congé de maternité."
        generated = "Le système de retraite au Maroc est géré par la CNSS et la CIMR."
        score = compute_faithfulness(generated, context)
        assert score < 0.5

    def test_empty_generated(self):
        """Score = 0 avec une réponse vide."""
        assert compute_faithfulness("", "Contexte") == 0.0

    def test_empty_context(self):
        """Score = 0 avec un contexte vide."""
        assert compute_faithfulness("Réponse", "") == 0.0


# ═══════════════════════════════════════════════════════════════════
# Tests de la métrique d'abstention
# ═══════════════════════════════════════════════════════════════════

class TestAbstentionAccuracy:
    """Tests pour la précision de l'abstention."""

    def test_perfect_abstention(self):
        """100% de précision quand toutes les décisions sont correctes."""
        predictions = [
            {"abstained": False},  # Devrait répondre → OK
            {"abstained": True},   # Devrait s'abstenir → OK
        ]
        ground_truth = [
            {"expected_sources": ["Source 1"]},  # Réponse attendue
            {"expected_sources": []},            # Abstention attendue
        ]
        assert abstention_accuracy(predictions, ground_truth) == 1.0

    def test_zero_abstention_accuracy(self):
        """0% de précision quand toutes les décisions sont incorrectes."""
        predictions = [
            {"abstained": True},   # S'abstient à tort
            {"abstained": False},  # Répond à tort
        ]
        ground_truth = [
            {"expected_sources": ["Source 1"]},  # Devrait répondre
            {"expected_sources": []},            # Devrait s'abstenir
        ]
        assert abstention_accuracy(predictions, ground_truth) == 0.0

    def test_partial_abstention_accuracy(self):
        """50% de précision (1/2 correct)."""
        predictions = [
            {"abstained": False},  # OK
            {"abstained": False},  # Mauvais (devrait s'abstenir)
        ]
        ground_truth = [
            {"expected_sources": ["Source 1"]},
            {"expected_sources": []},
        ]
        assert abstention_accuracy(predictions, ground_truth) == 0.5

    def test_empty_predictions(self):
        """Précision = 0.0 avec des prédictions vides."""
        assert abstention_accuracy([], [{"expected_sources": ["A"]}]) == 0.0

    def test_empty_ground_truth(self):
        """Précision = 0.0 avec un ground truth vide."""
        assert abstention_accuracy([{"abstained": False}], []) == 0.0


# ═══════════════════════════════════════════════════════════════════
# Tests d'intégration des métriques
# ═══════════════════════════════════════════════════════════════════

class TestMetricsIntegration:
    """Tests d'intégration combinant plusieurs métriques."""

    def test_all_metrics_return_float(self):
        """Vérifie que toutes les métriques retournent un float."""
        retrieved = ["A", "B"]
        relevant = ["A"]

        assert isinstance(precision_at_k(retrieved, relevant, 1), float)
        assert isinstance(recall_at_k(retrieved, relevant, 1), float)
        assert isinstance(mean_reciprocal_rank(retrieved, relevant), float)
        assert isinstance(rouge_l_score("texte", "texte"), float)
        assert isinstance(compute_faithfulness("texte", "contexte"), float)

    def test_metrics_bounded_0_to_1(self):
        """Vérifie que toutes les métriques sont entre 0 et 1."""
        retrieved = ["A", "B", "C"]
        relevant = ["A", "D"]

        metrics = [
            precision_at_k(retrieved, relevant, 3),
            recall_at_k(retrieved, relevant, 3),
            mean_reciprocal_rank(retrieved, relevant),
            rouge_l_score("texte test", "texte référence"),
            compute_faithfulness("réponse", "contexte"),
        ]

        for m in metrics:
            assert 0.0 <= m <= 1.0, f"Métrique hors bornes : {m}"
