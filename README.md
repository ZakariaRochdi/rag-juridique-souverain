# ⚖️ RAG Juridique Souverain

**Assistant juridique souverain pour le citoyen marocain** — un système de Recherche Augmentée (RAG) sur le corpus de droit marocain.

> Le citoyen pose une question en langage naturel ; le système recherche le passage pertinent dans le corpus juridique marocain, puis formule une réponse claire en citant systématiquement la source officielle.

---

## 🎯 Objectif

Concevoir un assistant de recherche juridique **souverain** fondé sur l'architecture RAG :
- **Fiabilité** : ancrage dans les textes réels, citations systématiques
- **Souveraineté** : exécution 100% locale, aucune API étrangère
- **Transparence** : gestion explicite du « je ne sais pas »

## 💡 Exemple d'interaction

```
Citoyen : « J'ai droit à combien de jours de congé de maternité ? »

Assistant : Selon l'article 152 du Code du travail (Loi 65-99), la salariée bénéficie
d'un congé de maternité de 14 semaines. Le contrat de travail est suspendu, non rompu,
pendant cette période.

📌 Source : Code du travail, article 152.
⚠️ Réponse fournie à titre informatif ; pour une situation particulière, consultez un professionnel.
```

---

## 🏗️ Architecture

Le pipeline comprend **8 étapes** :

| Étape | Description | Module |
|-------|-------------|--------|
| 1 | Constitution du corpus juridique | `data/raw/` |
| 2 | Prétraitement du corpus | `corpus/` |
| 3 | Construction de la base de connaissances | `indexation/` |
| 4 | Recherche des informations pertinentes | `retrieval/` |
| 5 | Génération de la réponse | `generation/` |
| 6 | Déploiement du système | `ui/` |
| 7 | Évaluation | `evaluation/`, `tests/` |
| 8 | Documentation et livrables | `docs/`, `rapport_stage/` |

---

## 📁 Structure du projet

```
rag-juridique-souverain/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py
├── data/
│   ├── raw/                  # Documents juridiques bruts (PDF)
│   ├── processed/            # Corpus nettoyé et découpé
│   ├── vector_store/         # Index FAISS + métadonnées
│   └── qa_eval/              # Jeu de questions-réponses d'évaluation
├── corpus/
│   ├── extraction.py         # Extraction de texte depuis PDF
│   ├── nettoyage.py          # Nettoyage et normalisation
│   └── decoupage.py          # Découpage par article/section
├── indexation/
│   └── build_index.py        # Embeddings + index FAISS
├── retrieval/
│   └── retriever.py          # Recherche sémantique
├── generation/
│   └── generateur.py         # Génération de réponses
├── evaluation/
│   ├── metrics.py            # Métriques d'évaluation
│   └── run_eval.py           # Pipeline d'évaluation
├── ui/
│   └── app.py                # Interface Streamlit
├── tests/
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_evaluation.py
├── docs/
│   ├── architecture.md
│   └── methodology.md
├── logs/
└── rapport_stage/
```

---

## ⚙️ Installation

### Prérequis

- Python 3.10+
- [Ollama](https://ollama.ai) installé localement (pour le LLM)

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-repo/rag-juridique-souverain.git
cd rag-juridique-souverain

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le modèle LLM via Ollama
ollama pull mistral

# 5. Placer les PDF juridiques dans data/raw/
cp chemin/vers/code-travail-maroc.pdf data/raw/
```

---

## 🚀 Utilisation

### Pipeline complet

```bash
# Ingestion : extraction + nettoyage + découpage
python main.py ingest

# Indexation : construction de la base vectorielle
python main.py index

# Requête interactive
python main.py query "Quelle est la durée du congé de maternité ?"

# Évaluation des performances
python main.py evaluate

# Lancer l'interface web Streamlit
python main.py serve
```

### Interface Streamlit directe

```bash
streamlit run ui/app.py
```

---

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest tests/ -v

# Tests spécifiques
pytest tests/test_retrieval.py -v
pytest tests/test_generation.py -v
pytest tests/test_evaluation.py -v
```

---

## 📊 Évaluation

Le système est évalué sur :
- **Qualité du retrieval** : Precision@K, Recall@K, MRR
- **Qualité de la génération** : ROUGE-L, fidélité aux sources
- **Performances globales** : taux de réponses correctes, taux d'abstention correcte

---

## 🔧 Stack technique

| Composant | Technologie |
|-----------|-------------|
| Embeddings | `sentence-transformers` (multilingual-e5-base) |
| Base vectorielle | FAISS |
| LLM | Ollama (Mistral 7B) |
| Extraction PDF | PyMuPDF |
| Interface | Streamlit |
| Tests | pytest |

---

## 📜 Licence

Ce projet est développé dans le cadre d'un stage chez **Pulsaride Solutions**.

## 👤 Auteur

- **Encadrant** : M. SALIHI Lyazid — Pulsaride Solutions
- **Contact** : contact@pulsaride.com
