"""
evaluation/run_eval.py — Pipeline d'évaluation complet du système RAG
======================================================================
Étape 7 du pipeline : Exécution de l'évaluation.

Ce script :
  1. Charge le jeu de questions-réponses de référence (qa_eval/)
  2. Exécute le pipeline RAG (retrieval + generation) sur chaque question
  3. Calcule les métriques d'évaluation
  4. Génère un rapport de résultats
"""

import json
from pathlib import Path
from typing import Dict, Any, List

from loguru import logger

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    rouge_l_score,
    compute_faithfulness,
    abstention_accuracy,
)


def load_qa_dataset(qa_path: str) -> List[Dict[str, Any]]:
    """
    Charge le jeu de questions-réponses de référence.

    Args:
        qa_path: Chemin vers le répertoire data/qa_eval/.

    Returns:
        Liste de questions de référence avec réponses attendues.
    """
    dataset_path = Path(qa_path) / "qa_dataset.json"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Jeu de données d'évaluation introuvable : {dataset_path}"
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"📂 {len(dataset)} question(s) de référence chargée(s)")
    return dataset


def evaluate_retrieval(
    retriever,
    dataset: List[Dict[str, Any]],
    k_values: List[int],
) -> Dict[str, float]:
    """
    Évalue la qualité du retrieval sur le jeu de données.

    Args:
        retriever: Instance de SemanticRetriever.
        dataset: Jeu de questions de référence.
        k_values: Liste de valeurs K pour Precision@K et Recall@K.

    Returns:
        Dictionnaire des métriques moyennes.
    """
    logger.info("🔍 Évaluation du retrieval...")

    metrics = {}

    for k in k_values:
        precisions = []
        recalls = []

        for qa in dataset:
            question = qa["question"]
            expected_sources = qa.get("expected_sources", [])

            if not expected_sources:
                continue  # Ignorer les questions d'abstention

            # Exécuter la recherche
            results = retriever.search(question, top_k=k)
            retrieved_sources = []
            for r in results:
                source = r["metadata"].get("source", "")
                article = r["metadata"].get("article", "")
                retrieved_sources.append(f"{source}, {article}")

            # Calculer les métriques
            p = precision_at_k(retrieved_sources, expected_sources, k)
            r = recall_at_k(retrieved_sources, expected_sources, k)

            precisions.append(p)
            recalls.append(r)

        if precisions:
            metrics[f"precision_at_{k}"] = sum(precisions) / len(precisions)
            metrics[f"recall_at_{k}"] = sum(recalls) / len(recalls)

    # MRR
    mrr_scores = []
    for qa in dataset:
        if not qa.get("expected_sources"):
            continue

        results = retriever.search(qa["question"])
        retrieved_sources = []
        for r in results:
            source = r["metadata"].get("source", "")
            article = r["metadata"].get("article", "")
            retrieved_sources.append(f"{source}, {article}")

        mrr = mean_reciprocal_rank(retrieved_sources, qa["expected_sources"])
        mrr_scores.append(mrr)

    if mrr_scores:
        metrics["mrr"] = sum(mrr_scores) / len(mrr_scores)

    logger.info(f"   → Métriques de retrieval calculées : {len(metrics)} métriques")
    return metrics


def evaluate_generation(
    retriever,
    generator,
    dataset: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Évalue la qualité de la génération sur le jeu de données.

    Args:
        retriever: Instance de SemanticRetriever.
        generator: Instance de ResponseGenerator.
        dataset: Jeu de questions de référence.

    Returns:
        Dictionnaire des métriques moyennes.
    """
    logger.info("📝 Évaluation de la génération...")

    rouge_scores = []
    faithfulness_scores = []
    predictions = []

    for qa in dataset:
        question = qa["question"]
        expected_answer = qa.get("expected_answer", "")

        # Pipeline RAG complet
        results = retriever.search(question)
        response = generator.generate(question, results)

        predictions.append(response)

        if not response.get("abstained") and expected_answer:
            # ROUGE-L
            rouge = rouge_l_score(response["answer"], expected_answer)
            rouge_scores.append(rouge)

            # Faithfulness
            context = "\n".join(r["text"] for r in results)
            faith = compute_faithfulness(response["answer"], context)
            faithfulness_scores.append(faith)

    metrics = {}

    if rouge_scores:
        metrics["rouge_l"] = sum(rouge_scores) / len(rouge_scores)

    if faithfulness_scores:
        metrics["faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores)

    # Abstention accuracy
    abs_acc = abstention_accuracy(predictions, dataset)
    metrics["abstention_accuracy"] = abs_acc

    logger.info(f"   → Métriques de génération calculées : {len(metrics)} métriques")
    return metrics


def run_evaluation(config: dict) -> Dict[str, float]:
    """
    Exécute le pipeline d'évaluation complet.

    Args:
        config: Configuration globale du projet.

    Returns:
        Dictionnaire de toutes les métriques.
    """
    from retrieval.retriever import SemanticRetriever
    from generation.generateur import ResponseGenerator

    logger.info("=" * 60)
    logger.info("🚀 Début de l'évaluation du système RAG")
    logger.info("=" * 60)

    # Charger le jeu de données
    dataset = load_qa_dataset(config["paths"]["qa_eval"])

    # Initialiser les composants
    retriever = SemanticRetriever(config)
    generator = ResponseGenerator(config)

    # Paramètres d'évaluation
    eval_config = config.get("evaluation", {})
    k_values = eval_config.get("k_values", [1, 3, 5])

    # Évaluation du retrieval
    retrieval_metrics = evaluate_retrieval(retriever, dataset, k_values)

    # Évaluation de la génération
    generation_metrics = evaluate_generation(retriever, generator, dataset)

    # Fusionner les résultats
    all_metrics = {**retrieval_metrics, **generation_metrics}

    # Log des résultats
    logger.info("\n📊 Résultats de l'évaluation :")
    for metric, value in sorted(all_metrics.items()):
        logger.info(f"   {metric}: {value:.4f}")

    # Sauvegarder les résultats
    results_path = Path(config["paths"]["qa_eval"]) / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.success(f"✅ Résultats sauvegardés dans {results_path}")

    return all_metrics


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results = run_evaluation(config)
    print("\n📊 Résultats :")
    for k, v in results.items():
        print(f"   {k}: {v:.4f}")
