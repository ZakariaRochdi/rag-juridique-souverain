"""
evaluation/metrics.py — Métriques d'évaluation du système RAG
==============================================================
Étape 7 du pipeline : Évaluation.

Métriques implémentées :
  - Retrieval : Precision@K, Recall@K, MRR (Mean Reciprocal Rank)
  - Génération : ROUGE-L, Faithfulness (fidélité aux sources)
  - Système : Taux d'abstention correcte
"""

import re
from typing import List, Set, Optional

from loguru import logger


# ─── Métriques de Retrieval ─────────────────────────────────────────

def precision_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int,
) -> float:
    """
    Calcule la Precision@K pour le retrieval.

    Precision@K = |résultats pertinents dans top-K| / K

    Args:
        retrieved_sources: Sources retournées par le retriever.
        relevant_sources: Sources pertinentes attendues (ground truth).
        k: Nombre de résultats à considérer.

    Returns:
        Score de précision entre 0 et 1.
    """
    if k == 0:
        return 0.0

    top_k = retrieved_sources[:k]
    relevant_set = set(_normalize_source(s) for s in relevant_sources)
    retrieved_set = set(_normalize_source(s) for s in top_k)

    relevant_retrieved = relevant_set & retrieved_set
    return len(relevant_retrieved) / k


def recall_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int,
) -> float:
    """
    Calcule le Recall@K pour le retrieval.

    Recall@K = |résultats pertinents dans top-K| / |résultats pertinents totaux|

    Args:
        retrieved_sources: Sources retournées par le retriever.
        relevant_sources: Sources pertinentes attendues.
        k: Nombre de résultats à considérer.

    Returns:
        Score de recall entre 0 et 1.
    """
    if not relevant_sources:
        return 0.0

    top_k = retrieved_sources[:k]
    relevant_set = set(_normalize_source(s) for s in relevant_sources)
    retrieved_set = set(_normalize_source(s) for s in top_k)

    relevant_retrieved = relevant_set & retrieved_set
    return len(relevant_retrieved) / len(relevant_set)


def mean_reciprocal_rank(
    retrieved_sources: List[str],
    relevant_sources: List[str],
) -> float:
    """
    Calcule le Mean Reciprocal Rank (MRR).

    MRR = 1 / rang du premier résultat pertinent

    Args:
        retrieved_sources: Sources retournées par le retriever (ordonnées).
        relevant_sources: Sources pertinentes attendues.

    Returns:
        Score MRR entre 0 et 1.
    """
    relevant_set = set(_normalize_source(s) for s in relevant_sources)

    for i, source in enumerate(retrieved_sources):
        if _normalize_source(source) in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def _normalize_source(source: str) -> str:
    """Normalise une référence de source pour la comparaison."""
    # Supprimer les espaces superflus, mettre en minuscules
    normalized = source.strip().lower()
    # Normaliser les numéros d'articles
    normalized = re.sub(r"articles?\s*", "article ", normalized)
    # Supprimer les extensions de fichier
    normalized = re.sub(r"\.pdf", "", normalized)
    return normalized


# ─── Métriques de Génération ────────────────────────────────────────

def rouge_l_score(generated: str, reference: str) -> float:
    """
    Calcule le score ROUGE-L (Longest Common Subsequence).

    Mesure le chevauchement de la plus longue sous-séquence commune
    entre la réponse générée et la réponse de référence.

    Args:
        generated: Réponse générée par le système.
        reference: Réponse de référence attendue.

    Returns:
        Score ROUGE-L F1 entre 0 et 1.
    """
    if not generated or not reference:
        return 0.0

    # Tokenisation simple (mots)
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    if not gen_tokens or not ref_tokens:
        return 0.0

    # Calcul de la LCS
    lcs_length = _lcs_length(gen_tokens, ref_tokens)

    # Precision et Recall
    precision = lcs_length / len(gen_tokens) if gen_tokens else 0.0
    recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0

    # F1 score
    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Calcule la longueur de la plus longue sous-séquence commune (LCS)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def compute_faithfulness(
    generated: str,
    context: str,
) -> float:
    """
    Calcule un score de fidélité (faithfulness) de la réponse au contexte.

    Mesure la proportion de mots significatifs de la réponse qui
    apparaissent dans le contexte fourni.

    Args:
        generated: Réponse générée par le système.
        context: Contexte agrégé fourni au LLM.

    Returns:
        Score de fidélité entre 0 et 1.
    """
    if not generated or not context:
        return 0.0

    # Mots à ignorer (stop words français basiques)
    stop_words = {
        "le", "la", "les", "un", "une", "des", "de", "du", "d",
        "et", "ou", "en", "à", "au", "aux", "par", "pour", "dans",
        "sur", "avec", "ce", "cette", "ces", "est", "sont", "a",
        "ont", "qui", "que", "il", "elle", "ils", "elles", "ne",
        "pas", "se", "sa", "son", "ses", "nous", "vous", "leur",
        "leurs", "être", "avoir", "faire", "dit", "peut", "plus",
        "tout", "tous", "toute", "toutes", "même", "aussi",
        "l", "n", "s", "qu", "c",
    }

    # Extraire les mots significatifs de la réponse
    gen_words = set(
        w.lower()
        for w in re.findall(r"\b\w+\b", generated)
        if w.lower() not in stop_words and len(w) > 2
    )

    # Mots du contexte
    context_words = set(
        w.lower()
        for w in re.findall(r"\b\w+\b", context)
    )

    if not gen_words:
        return 0.0

    # Proportion de mots de la réponse présents dans le contexte
    faithful_words = gen_words & context_words
    return len(faithful_words) / len(gen_words)


# ─── Métriques Système ──────────────────────────────────────────────

def abstention_accuracy(
    predictions: List[dict],
    ground_truth: List[dict],
) -> float:
    """
    Calcule la précision de l'abstention.

    Mesure si le système s'abstient correctement quand il ne devrait
    pas répondre, et répond quand il le devrait.

    Args:
        predictions: Liste de réponses du système avec clé 'abstained'.
        ground_truth: Liste de questions de référence avec
                      clé 'expected_sources' (vide = abstention attendue).

    Returns:
        Taux de décisions d'abstention correctes.
    """
    if not predictions or not ground_truth:
        return 0.0

    correct = 0
    total = min(len(predictions), len(ground_truth))

    for pred, truth in zip(predictions, ground_truth):
        should_abstain = len(truth.get("expected_sources", [])) == 0
        did_abstain = pred.get("abstained", False)

        if should_abstain == did_abstain:
            correct += 1

    return correct / total


if __name__ == "__main__":
    # Tests rapides
    print("=== Tests des métriques ===\n")

    # Precision@K
    retrieved = ["Code du travail, Article 152", "Code du travail, Article 153"]
    relevant = ["Code du travail, Article 152"]
    print(f"Precision@1: {precision_at_k(retrieved, relevant, 1):.2f}")
    print(f"Precision@2: {precision_at_k(retrieved, relevant, 2):.2f}")

    # Recall@K
    print(f"Recall@1: {recall_at_k(retrieved, relevant, 1):.2f}")
    print(f"Recall@2: {recall_at_k(retrieved, relevant, 2):.2f}")

    # MRR
    print(f"MRR: {mean_reciprocal_rank(retrieved, relevant):.2f}")

    # ROUGE-L
    gen = "La salariée a droit à un congé de maternité de 14 semaines."
    ref = "La salariée bénéficie d'un congé de maternité de quatorze semaines."
    print(f"\nROUGE-L: {rouge_l_score(gen, ref):.4f}")

    # Faithfulness
    ctx = "Article 152 : La salariée a droit à un congé de maternité de quatorze semaines."
    print(f"Faithfulness: {compute_faithfulness(gen, ctx):.4f}")
