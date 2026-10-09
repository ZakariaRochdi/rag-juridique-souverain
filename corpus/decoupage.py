"""
corpus/decoupage.py — Découpage du corpus juridique en chunks
==============================================================
Étape 2b du pipeline : Prétraitement du corpus.

Stratégies de découpage :
  - "article" : découpage par article juridique (regex Article \\d+)
  - "fixed_size" : découpage en blocs de taille fixe avec chevauchement
  - "sentence" : découpage par phrases (via séparateurs)

Chaque chunk contient :
  - text : le texte du chunk
  - metadata : source, article, position, estimation de tokens
  - id : identifiant unique du chunk
"""

import re
from typing import List, Dict, Any

from loguru import logger


def _estimate_tokens(text: str) -> int:
    """
    Estime le nombre de tokens dans un texte.

    Approximation : 1 token ≈ 4 caractères en français.
    Plus précis qu'un simple split par espaces.

    Args:
        text: Texte à estimer.

    Returns:
        Estimation du nombre de tokens.
    """
    return max(1, len(text) // 4)


def _chunk_by_article(
    text: str,
    source: str,
    min_chunk_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Découpe un texte juridique par article.

    Détecte les motifs "Article \\d+" et crée un chunk pour chaque article.
    Les articles trop courts sont fusionnés avec le suivant.

    Args:
        text: Texte nettoyé du document.
        source: Nom du fichier source.
        min_chunk_size: Taille minimale d'un chunk en caractères.

    Returns:
        Liste de chunks avec texte et métadonnées.
    """
    # Pattern pour détecter les articles
    # Capture "Article 152", "Article 152 :", "Article 152.", "Article premier"
    article_pattern = re.compile(
        r"(?=(?:^|\n)\s*(Article\s+\d+[\s\S]*?)(?=\n\s*Article\s+\d+|\Z))",
        re.IGNORECASE,
    )

    # Alternative : split direct sur "Article \d+"
    # On utilise re.split pour diviser le texte aux frontières d'articles
    parts = re.split(
        r"(?=\bArticle\s+\d+)",
        text,
        flags=re.IGNORECASE,
    )

    chunks = []
    # Le premier élément avant tout article (préambule, titres...)
    preamble = parts[0].strip() if parts else ""

    if preamble and len(preamble) >= min_chunk_size:
        chunks.append({
            "text": preamble,
            "metadata": {
                "source": source,
                "article": "Préambule",
                "section": "introduction",
            },
        })

    # Traiter chaque article
    current_text = ""
    current_article = ""

    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue

        # Extraire le numéro d'article
        match = re.match(r"(Article\s+\d+)", part, re.IGNORECASE)
        article_ref = match.group(1) if match else "Article inconnu"

        if current_text and len(current_text) >= min_chunk_size:
            # Sauvegarder le chunk précédent
            chunks.append({
                "text": current_text,
                "metadata": {
                    "source": source,
                    "article": current_article,
                    "section": _detect_section(current_text),
                },
            })
            current_text = part
            current_article = article_ref
        elif current_text:
            # Fusionner avec le chunk précédent (trop court)
            current_text += "\n\n" + part
            # Garder la référence du premier article du groupe
        else:
            current_text = part
            current_article = article_ref

    # Dernier chunk
    if current_text:
        chunks.append({
            "text": current_text,
            "metadata": {
                "source": source,
                "article": current_article,
                "section": _detect_section(current_text),
            },
        })

    return chunks


def _chunk_by_fixed_size(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Découpe un texte en blocs de taille fixe avec chevauchement.

    Args:
        text: Texte nettoyé du document.
        source: Nom du fichier source.
        chunk_size: Taille cible en tokens.
        chunk_overlap: Chevauchement entre chunks en tokens.
        min_chunk_size: Taille minimale d'un chunk en tokens.

    Returns:
        Liste de chunks avec texte et métadonnées.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        if _estimate_tokens(chunk_text) >= min_chunk_size or start + chunk_size >= len(words):
            # Détecter l'article le plus proche
            article_ref = _find_nearest_article(chunk_text)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "article": article_ref,
                    "section": _detect_section(chunk_text),
                    "chunk_start_word": start,
                },
            })

        start += chunk_size - chunk_overlap

    return chunks


def _chunk_by_separators(
    text: str,
    source: str,
    separators: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100,
) -> List[Dict[str, Any]]:
    """
    Découpe un texte en utilisant une hiérarchie de séparateurs.

    Essaie les séparateurs dans l'ordre (du plus fort au plus faible)
    jusqu'à obtenir des chunks de taille appropriée.

    Args:
        text: Texte nettoyé.
        source: Nom du fichier source.
        separators: Liste ordonnée de séparateurs.
        chunk_size: Taille cible en caractères.
        chunk_overlap: Chevauchement en caractères.
        min_chunk_size: Taille minimale.

    Returns:
        Liste de chunks.
    """
    chunks = []
    current_chunk = ""

    # Utiliser le premier séparateur qui divise le texte
    split_text = [text]
    for sep in separators:
        new_splits = []
        for segment in split_text:
            parts = segment.split(sep)
            for i, part in enumerate(parts):
                if i > 0:
                    new_splits.append(sep + part)
                else:
                    new_splits.append(part)
        if len(new_splits) > len(split_text):
            split_text = new_splits
            break

    for part in split_text:
        part = part.strip()
        if not part:
            continue

        if len(current_chunk) + len(part) <= chunk_size * 4:  # ~chunk_size tokens
            current_chunk += "\n" + part if current_chunk else part
        else:
            if current_chunk and len(current_chunk) >= min_chunk_size:
                article_ref = _find_nearest_article(current_chunk)
                chunks.append({
                    "text": current_chunk,
                    "metadata": {
                        "source": source,
                        "article": article_ref,
                        "section": _detect_section(current_chunk),
                    },
                })
            current_chunk = part

    # Dernier chunk
    if current_chunk and len(current_chunk) >= min_chunk_size:
        article_ref = _find_nearest_article(current_chunk)
        chunks.append({
            "text": current_chunk,
            "metadata": {
                "source": source,
                "article": article_ref,
                "section": _detect_section(current_chunk),
            },
        })

    return chunks


def _find_nearest_article(text: str) -> str:
    """Trouve la référence d'article la plus proche dans un texte."""
    match = re.search(r"(Article\s+\d+)", text, re.IGNORECASE)
    return match.group(1) if match else ""


def _detect_section(text: str) -> str:
    """Détecte la section/chapitre/titre à partir du texte."""
    # Chercher les titres de sections
    for pattern, label in [
        (r"Titre\s+(\w+)", "titre"),
        (r"Chapitre\s+(\w+)", "chapitre"),
        (r"Section\s+(\w+)", "section"),
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{label}_{match.group(1).lower()}"
    return ""


def chunk_all_documents(
    documents: List[Dict[str, Any]],
    chunking_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Découpe tous les documents en chunks selon la stratégie configurée.

    Args:
        documents: Liste de documents nettoyés (clé 'text' et 'metadata').
        chunking_config: Configuration du découpage depuis config.yaml :
            - strategy: "article", "fixed_size", ou "sentence"
            - chunk_size: taille cible en tokens
            - chunk_overlap: chevauchement en tokens
            - min_chunk_size: taille minimale en tokens
            - separators: séparateurs pour la stratégie "sentence"

    Returns:
        Liste complète de chunks avec identifiants uniques.
    """
    strategy = chunking_config.get("strategy", "article")
    chunk_size = chunking_config.get("chunk_size", 500)
    chunk_overlap = chunking_config.get("chunk_overlap", 50)
    min_chunk_size = chunking_config.get("min_chunk_size", 100)
    separators = chunking_config.get("separators", ["\n\n", "\n"])

    logger.info(
        f"✂️ Découpage avec stratégie '{strategy}' "
        f"(taille={chunk_size}, overlap={chunk_overlap})"
    )

    all_chunks = []
    chunk_id = 0

    for doc in documents:
        text = doc["text"]
        source = doc["metadata"].get("source", "inconnu")

        logger.info(f"   📄 Découpage de : {source}")

        if strategy == "article":
            doc_chunks = _chunk_by_article(text, source, min_chunk_size)
        elif strategy == "fixed_size":
            doc_chunks = _chunk_by_fixed_size(
                text, source, chunk_size, chunk_overlap, min_chunk_size
            )
        elif strategy == "sentence":
            doc_chunks = _chunk_by_separators(
                text, source, separators, chunk_size, chunk_overlap, min_chunk_size
            )
        else:
            logger.warning(
                f"   Stratégie inconnue '{strategy}', "
                "utilisation de 'article' par défaut"
            )
            doc_chunks = _chunk_by_article(text, source, min_chunk_size)

        # Ajouter l'ID et l'estimation de tokens à chaque chunk
        for chunk in doc_chunks:
            chunk["id"] = chunk_id
            chunk["metadata"]["tokens_estimate"] = _estimate_tokens(chunk["text"])
            chunk_id += 1

        logger.info(f"      → {len(doc_chunks)} chunk(s) créé(s)")
        all_chunks.extend(doc_chunks)

    logger.info(
        f"   → Total : {len(all_chunks)} chunk(s) pour "
        f"{len(documents)} document(s)"
    )

    return all_chunks


if __name__ == "__main__":
    # Test rapide
    sample_docs = [
        {
            "text": (
                "Titre premier\nDispositions générales\n\n"
                "Article 1 : Les dispositions de la présente loi "
                "s'appliquent aux personnes liées par un contrat de travail.\n\n"
                "Article 2 : Les dispositions de la présente loi s'appliquent "
                "également aux salariés du secteur public.\n\n"
                "Article 3 : Sont exclus du champ d'application les "
                "fonctionnaires de l'État.\n\n"
                "Chapitre premier\n\n"
                "Article 152 : La salariée a droit à un congé de maternité "
                "de quatorze semaines. Le contrat de travail est suspendu "
                "pendant cette période.\n\n"
                "Article 153 : La salariée peut demander une prolongation "
                "de congé non payé d'une durée maximale de 90 jours.\n"
            ),
            "metadata": {"source": "code-travail-maroc.pdf"},
        }
    ]

    config = {
        "strategy": "article",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "min_chunk_size": 50,
    }

    chunks = chunk_all_documents(sample_docs, config)

    for c in chunks:
        print(f"\n📎 ID={c['id']} | {c['metadata']['article']} "
              f"| ~{c['metadata']['tokens_estimate']} tokens")
        print(f"   {c['text'][:120]}...")
