"""
RAG Juridique Souverain — Point d'entrée principal
===================================================
Ce script orchestre l'ensemble du pipeline RAG :
  - ingest   : extraction, nettoyage et découpage du corpus
  - index    : construction de la base vectorielle (FAISS)
  - query    : interrogation du système en ligne de commande
  - evaluate : évaluation des performances du système
  - serve    : lancement de l'interface Streamlit
"""

import argparse
import sys
import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import yaml
from pathlib import Path
from loguru import logger



# ─── Configuration du logging ───────────────────────────────────────
def setup_logging(config: dict) -> None:
    """Configure le système de logging avec loguru."""
    log_config = config.get("logging", {})
    log_dir = Path(config["paths"]["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Supprime le handler par défaut
    logger.add(
        sys.stderr,
        level=log_config.get("level", "INFO"),
        format=log_config.get(
            "format",
            "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        ),
    )
    logger.add(
        log_dir / "app.log",
        level=log_config.get("level", "INFO"),
        format=log_config.get(
            "format",
            "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
        ),
        rotation=log_config.get("rotation", "10 MB"),
        retention=log_config.get("retention", "30 days"),
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """Charge la configuration depuis le fichier YAML."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Fichier de configuration introuvable : {config_path}")
        sys.exit(1)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration chargée depuis {config_path}")
    return config


# ─── Commandes du pipeline ──────────────────────────────────────────
def cmd_ingest(config: dict) -> None:
    """Étapes 1-2 : Extraction, nettoyage et découpage du corpus."""
    from corpus.extraction import extract_all_documents
    from corpus.nettoyage import clean_all_documents
    from corpus.decoupage import chunk_all_documents

    logger.info("=" * 60)
    logger.info("ÉTAPE 1-2 : Ingestion du corpus juridique")
    logger.info("=" * 60)

    # Création des répertoires nécessaires
    Path(config["paths"]["processed_data"]).mkdir(parents=True, exist_ok=True)

    # Extraction
    logger.info("📄 Extraction du texte depuis les PDF...")
    raw_texts = extract_all_documents(config["paths"]["raw_data"])
    logger.info(f"   → {len(raw_texts)} document(s) extrait(s)")

    # Nettoyage
    logger.info("🧹 Nettoyage et normalisation...")
    cleaned_texts = clean_all_documents(raw_texts)
    logger.info(f"   → {len(cleaned_texts)} document(s) nettoyé(s)")

    # Découpage
    logger.info("✂️ Découpage en chunks...")
    chunks = chunk_all_documents(cleaned_texts, config["chunking"])
    logger.info(f"   → {len(chunks)} chunk(s) créé(s)")

    # Sauvegarde
    import json

    output_path = Path(config["paths"]["processed_data"]) / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.success(f"✅ Corpus ingéré et sauvegardé dans {output_path}")


def cmd_index(config: dict) -> None:
    """Étape 3 : Construction de la base vectorielle FAISS."""
    from indexation.build_index import build_vector_index

    logger.info("=" * 60)
    logger.info("ÉTAPE 3 : Construction de la base de connaissances")
    logger.info("=" * 60)

    # Création du répertoire
    Path(config["paths"]["vector_store"]).mkdir(parents=True, exist_ok=True)

    build_vector_index(config)
    logger.success("✅ Index vectoriel construit avec succès")


def cmd_query(config: dict, question: str) -> None:
    """Étapes 4-5 : Recherche + génération de réponse."""
    from retrieval.retriever import SemanticRetriever
    from generation.generateur import ResponseGenerator

    logger.info("=" * 60)
    logger.info("ÉTAPES 4-5 : Recherche et génération")
    logger.info("=" * 60)
    logger.info(f"❓ Question : {question}")

    # Recherche des passages pertinents
    retriever = SemanticRetriever(config)
    results = retriever.search(question)

    if not results:
        print("\n⚠️ Aucun passage pertinent trouvé dans le corpus.")
        return

    logger.info(f"   → {len(results)} passage(s) pertinent(s) trouvé(s)")

    # Génération de la réponse
    generator = ResponseGenerator(config)
    response = generator.generate(question, results)

    # Affichage
    print("\n" + "=" * 60)
    print("📋 RÉPONSE")
    print("=" * 60)
    print(response["answer"])
    print("\n📌 SOURCES :")
    for source in response.get("sources", []):
        print(f"   - {source}")
    print(
        "\n⚠️ Réponse fournie à titre informatif ; "
        "pour une situation particulière, consultez un professionnel."
    )
    print("=" * 60)


def cmd_evaluate(config: dict) -> None:
    """Étape 7 : Évaluation des performances du système."""
    from evaluation.run_eval import run_evaluation

    logger.info("=" * 60)
    logger.info("ÉTAPE 7 : Évaluation du système")
    logger.info("=" * 60)

    results = run_evaluation(config)

    # Affichage des résultats
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DE L'ÉVALUATION")
    print("=" * 60)
    for metric, value in results.items():
        print(f"   {metric}: {value:.4f}")
    print("=" * 60)


def cmd_serve(config: dict) -> None:
    """Étape 6 : Lancement de l'interface Streamlit."""
    logger.info("=" * 60)
    logger.info("ÉTAPE 6 : Lancement de l'interface Streamlit")
    logger.info("=" * 60)

    os.system("streamlit run ui/app.py")


# ─── Point d'entrée ─────────────────────────────────────────────────
def main():
    """Point d'entrée principal du pipeline RAG Juridique Souverain."""
    parser = argparse.ArgumentParser(
        description="⚖️ RAG Juridique Souverain — Pipeline complet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
  python main.py ingest                          Ingérer le corpus juridique
  python main.py index                           Construire la base vectorielle
  python main.py query "Durée congé maternité ?" Poser une question
  python main.py evaluate                        Évaluer les performances
  python main.py serve                           Lancer l'interface web
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # Sous-commande : ingest
    subparsers.add_parser("ingest", help="Extraire, nettoyer et découper le corpus")

    # Sous-commande : index
    subparsers.add_parser("index", help="Construire la base vectorielle FAISS")

    # Sous-commande : query
    query_parser = subparsers.add_parser("query", help="Poser une question au système")
    query_parser.add_argument("question", type=str, help="Question en langage naturel")

    # Sous-commande : evaluate
    subparsers.add_parser("evaluate", help="Évaluer les performances du système")

    # Sous-commande : serve
    subparsers.add_parser("serve", help="Lancer l'interface Streamlit")

    # Argument global : config
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Chemin vers le fichier de configuration (défaut: config.yaml)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Chargement de la config et setup logging
    config = load_config(args.config)
    setup_logging(config)

    # Dispatch des commandes
    if args.command == "ingest":
        cmd_ingest(config)
    elif args.command == "index":
        cmd_index(config)
    elif args.command == "query":
        cmd_query(config, args.question)
    elif args.command == "evaluate":
        cmd_evaluate(config)
    elif args.command == "serve":
        cmd_serve(config)


if __name__ == "__main__":
    main()
