"""
corpus/nettoyage.py — Nettoyage et normalisation du corpus juridique
=====================================================================
Étape 2a du pipeline : Prétraitement du corpus.

Opérations de nettoyage :
  1. Suppression des caractères de contrôle et artefacts PDF
  2. Suppression de la table des matières (lignes de points)
  3. Suppression des en-têtes/pieds de page et numéros de page
  4. Normalisation de l'espacement et de la ponctuation
  5. Normalisation Unicode (accents, caractères spéciaux)
"""

import re
from typing import List, Dict, Any

from loguru import logger

try:
    from unidecode import unidecode  # noqa: F401 — vérifie la disponibilité
except ImportError:
    pass  # Optionnel, on continue sans


def clean_text(text: str) -> str:
    """
    Nettoie un texte brut extrait d'un PDF juridique.

    Applique une série de transformations pour retirer les artefacts
    d'extraction PDF et normaliser le texte pour le traitement NLP.

    Args:
        text: Texte brut extrait du PDF.

    Returns:
        Texte nettoyé et normalisé.
    """
    if not text:
        return ""

    cleaned = text

    # 1. Supprimer les caractères de contrôle (sauf \n et \t)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # 2. Supprimer les signets Word (_Toc...)
    cleaned = re.sub(r"_Toc\d+", "", cleaned)

    # 3. Supprimer les lignes de la table des matières
    #    (lignes contenant principalement des points ou des tirets)
    cleaned = re.sub(
        r"^.*[\.]{3,}.*$",
        "",
        cleaned,
        flags=re.MULTILINE,
    )
    # Variante avec des tirets ou underscores répétés
    cleaned = re.sub(
        r"^.*[-_]{10,}.*$",
        "",
        cleaned,
        flags=re.MULTILINE,
    )

    # 3. Supprimer les numéros de page isolés ou "Page X / Y"
    cleaned = re.sub(
        r"^\s*(\d{1,4}|Page\s+\d+(\s*/\s*\d+)?)\s*$",
        "",
        cleaned,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 4. Supprimer les en-têtes/pieds de page récurrents
    #    (ex: "Code du travail" seul sur une ligne, "Loi n° 65-99" seul)
    cleaned = re.sub(
        r"^\s*(Code du travail|Loi n°?\s*65[-–]99)\s*$",
        "",
        cleaned,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 5. Normaliser les espaces multiples (mais préserver les sauts de ligne)
    cleaned = re.sub(r"[^\S\n]+", " ", cleaned)

    # 6. Supprimer les lignes vides multiples (garder max 2 sauts de ligne)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 7. Normaliser les tirets et guillemets
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = cleaned.replace("«", '"').replace("»", '"')
    cleaned = cleaned.replace("\u00a0", " ")  # Espace insécable

    # 8. Nettoyer les espaces en début/fin de ligne
    lines = cleaned.split("\n")
    lines = [line.strip() for line in lines]
    cleaned = "\n".join(lines)

    # 9. Supprimer les espaces en début/fin du texte complet
    cleaned = cleaned.strip()

    return cleaned


def clean_all_documents(
    documents: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Nettoie tous les documents extraits.

    Args:
        documents: Liste de documents avec clé 'text' et optionnellement 'metadata'.

    Returns:
        Liste de documents avec le texte nettoyé.
        Les documents dont le texte nettoyé est vide sont exclus.
    """
    logger.info(f"🧹 Nettoyage de {len(documents)} document(s)...")

    cleaned_docs = []
    for doc in documents:
        original_length = len(doc["text"])
        cleaned_text = clean_text(doc["text"])
        cleaned_length = len(cleaned_text)
        metadata = dict(doc.get("metadata", {}))
        source = metadata.get("source", doc.get("filename", "inconnu"))
        metadata["source"] = source

        if not cleaned_text:
            logger.warning(
                f"   ⚠️ Document vide après nettoyage : {source}"
            )
            continue

        reduction = (1 - cleaned_length / original_length) * 100 if original_length > 0 else 0

        logger.debug(
            f"   📄 {source} : "
            f"{original_length} → {cleaned_length} caractères "
            f"(-{reduction:.1f}%)"
        )

        cleaned_docs.append({
            "text": cleaned_text,
            "metadata": {
                **metadata,
                "original_length": original_length,
                "cleaned_length": cleaned_length,
            },
        })

    logger.info(f"   → {len(cleaned_docs)} document(s) nettoyé(s)")
    return cleaned_docs


if __name__ == "__main__":
    # Test rapide avec un extrait
    sample = (
        "Code du travail\n"
        "\n"
        "Article 2 ..........................................1\x01\n"
        "Article 3 ..........................................2\n"
        "\n"
        "  42  \n"
        "\n"
        "Article 152 : La salariée a droit à un congé de\n"
        "maternité de quatorze  semaines.\n"
        "\n\n\n\n"
        "Loi n° 65-99\n"
        "Article 153 : La salariée peut demander une prolongation.\n"
    )

    result = clean_text(sample)
    print("=== Texte nettoyé ===")
    print(result)
    print(f"\n=== Réduction : {len(sample)} → {len(result)} caractères ===")
