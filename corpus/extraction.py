"""
corpus/extraction.py — Extraction de texte depuis les documents PDF
====================================================================
Étape 1 du pipeline : Constitution du corpus juridique.

Utilise PyMuPDF (fitz) pour extraire le texte brut de chaque page
des documents PDF juridiques.

Opérations :
  1. Parcourt tous les fichiers PDF dans data/raw/
  2. Extrait le texte page par page
  3. Retourne une liste de documents avec texte et métadonnées
"""

from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF est requis pour l'extraction PDF. "
        "Installez-le avec : pip install PyMuPDF"
    )


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """
    Extrait le texte d'un fichier PDF.

    Parcourt chaque page du document et concatène le texte extrait.
    Collecte également les métadonnées du document (nombre de pages,
    nom du fichier).

    Args:
        pdf_path: Chemin vers le fichier PDF.

    Returns:
        Dictionnaire contenant :
          - text: texte brut extrait du PDF
          - metadata: informations sur le document (source, pages)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Fichier PDF introuvable : {pdf_path}")

    logger.info(f"   📄 Extraction de : {pdf_path.name}")

    doc = fitz.open(str(pdf_path))
    num_pages = len(doc)
    pages_text = []

    for page_num in range(num_pages):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)

    doc.close()

    full_text = "\n\n".join(pages_text)

    logger.debug(
        f"      → {num_pages} page(s), "
        f"{len(full_text)} caractères extraits"
    )

    return {
        "text": full_text,
        "metadata": {
            "source": pdf_path.name,
            "num_pages": num_pages,
            "file_path": str(pdf_path),
        },
    }


def extract_all_documents(raw_data_path: str) -> List[Dict[str, Any]]:
    """
    Extrait le texte de tous les fichiers PDF dans le répertoire donné.

    Args:
        raw_data_path: Chemin vers le répertoire contenant les PDF
                       (ex: data/raw/).

    Returns:
        Liste de documents, chacun contenant le texte et les métadonnées.

    Raises:
        FileNotFoundError: Si le répertoire n'existe pas ou ne contient
                           aucun PDF.
    """
    raw_dir = Path(raw_data_path)

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Répertoire de données brutes introuvable : {raw_data_path}"
        )

    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Aucun fichier PDF trouvé dans : {raw_data_path}. "
            "Placez vos documents juridiques (PDF) dans ce répertoire."
        )

    logger.info(f"📂 {len(pdf_files)} fichier(s) PDF trouvé(s) dans {raw_data_path}")

    documents = []
    for pdf_path in pdf_files:
        try:
            doc = extract_text_from_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"   ❌ Erreur lors de l'extraction de {pdf_path.name} : {e}")

    logger.info(f"   → {len(documents)} document(s) extrait(s) avec succès")
    return documents


if __name__ == "__main__":
    # Test rapide
    docs = extract_all_documents("data/raw")
    for doc in docs:
        print(f"\n📄 {doc['metadata']['source']}")
        print(f"   Pages : {doc['metadata']['num_pages']}")
        print(f"   Caractères : {len(doc['text'])}")
        print(f"   Extrait : {doc['text'][:200]}...")
