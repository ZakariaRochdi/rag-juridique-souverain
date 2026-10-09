"""
ui/app.py — Interface Streamlit de l'assistant juridique souverain
===================================================================
Étape 6 du pipeline : Déploiement du système.

Interface web interactive permettant au citoyen de :
  - Poser des questions en langage naturel
  - Consulter les réponses sourcées
  - Voir les extraits juridiques pertinents
  - Naviguer dans l'historique des conversations
"""

import sys
from pathlib import Path

import streamlit as st
import yaml
from loguru import logger

# Ajouter le répertoire racine au path pour les imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from retrieval.retriever import SemanticRetriever
from generation.generateur import ResponseGenerator


# ─── Configuration de la page ──────────────────────────────────────
def load_app_config() -> dict:
    """Charge la configuration depuis config.yaml."""
    config_path = ROOT_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_page(config: dict) -> None:
    """Configure la page Streamlit."""
    ui_config = config.get("ui", {})

    st.set_page_config(
        page_title=ui_config.get("title", "Assistant Juridique Souverain"),
        page_icon=ui_config.get("page_icon", "⚖️"),
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ─── Styles CSS personnalisés ──────────────────────────────────────
def inject_css() -> None:
    """Injecte les styles CSS personnalisés."""
    st.markdown(
        """
        <style>
        /* Header principal */
        .main-header {
            text-align: center;
            padding: 1rem 0;
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            font-size: 2rem;
            margin: 0;
        }
        .main-header p {
            opacity: 0.8;
            margin: 0.5rem 0 0 0;
        }

        /* Carte de résultat */
        .result-card {
            background: #f8f9fa;
            border-left: 4px solid #2c5282;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
        }

        /* Source badge */
        .source-badge {
            display: inline-block;
            background: #2c5282;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.85rem;
            margin: 0.2rem;
        }

        /* Score bar */
        .score-bar {
            height: 6px;
            background: #e2e8f0;
            border-radius: 3px;
            margin-top: 0.5rem;
        }
        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
            border-radius: 3px;
        }

        /* Avertissement */
        .warning-box {
            background: #fffbeb;
            border: 1px solid #f6e05e;
            border-radius: 8px;
            padding: 0.8rem;
            margin-top: 1rem;
            font-size: 0.9rem;
        }

        /* Abstention */
        .abstention-box {
            background: #fff5f5;
            border: 1px solid #fc8181;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Composants d'affichage ────────────────────────────────────────
def display_header(config: dict) -> None:
    """Affiche l'en-tête de l'application."""
    ui_config = config.get("ui", {})
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{ui_config.get('title', '⚖️ Assistant Juridique Souverain')}</h1>
            <p>{ui_config.get('subtitle', 'Posez votre question sur le droit marocain')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_response(response: dict) -> None:
    """Affiche la réponse générée."""
    if response.get("abstained"):
        st.markdown(
            """
            <div class="abstention-box">
                <h3>🚫 Information non disponible</h3>
                <p>Je ne dispose pas de cette information dans les textes
                juridiques consultés. Je vous recommande de consulter un
                professionnel du droit.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Réponse principale
    st.markdown("### 📋 Réponse")
    st.write(response["answer"])

    # Sources
    if response.get("sources"):
        st.markdown("### 📌 Sources")
        sources_html = " ".join(
            f'<span class="source-badge">{s}</span>'
            for s in response["sources"]
        )
        st.markdown(sources_html, unsafe_allow_html=True)

    # Avertissement
    st.markdown(
        """
        <div class="warning-box">
            ⚠️ <strong>Avertissement :</strong> Réponse fournie à titre informatif ;
            pour une situation particulière, consultez un professionnel du droit.
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_sources_detail(results: list) -> None:
    """Affiche le détail des extraits trouvés dans un expander."""
    if not results:
        return

    with st.expander("📎 Voir les extraits juridiques détaillés", expanded=False):
        for i, r in enumerate(results):
            article = r["metadata"].get("article", "")
            source = r["metadata"].get("source", "")
            score = r.get("score", 0)

            st.markdown(
                f"""
                <div class="result-card">
                    <strong>📄 {source} — {article}</strong>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {score * 100:.0f}%"></div>
                    </div>
                    <small>Score de pertinence : {score:.2%}</small>
                    <p style="margin-top: 0.5rem;">{r['text'][:600]}{'...' if len(r['text']) > 600 else ''}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─── Sidebar ───────────────────────────────────────────────────────
def display_sidebar(config: dict) -> None:
    """Affiche la barre latérale avec les informations et paramètres."""
    with st.sidebar:
        st.markdown("## ⚙️ Paramètres")

        # Paramètres de recherche
        st.markdown("### 🔍 Recherche")
        top_k = st.slider(
            "Nombre de résultats (top-K)",
            min_value=1,
            max_value=10,
            value=config.get("retrieval", {}).get("top_k", 5),
            key="top_k",
        )

        threshold = st.slider(
            "Seuil de similarité",
            min_value=0.0,
            max_value=1.0,
            value=float(
                config.get("retrieval", {}).get("similarity_threshold", 0.3)
            ),
            step=0.05,
            key="threshold",
        )

        st.markdown("---")

        # Informations
        st.markdown("### ℹ️ À propos")
        st.markdown(
            """
            **RAG Juridique Souverain**

            Assistant de recherche juridique basé sur
            l'architecture RAG, conçu pour le droit marocain.

            🔒 **100% souverain** : toutes les données
            et le traitement restent locaux.

            📚 **Corpus** : Code du travail marocain
            (Loi 65-99)

            🤖 **Modèle** : Mistral 7B via Ollama
            """
        )

        st.markdown("---")
        st.markdown(
            "*Développé par [Rochdi Zakaria](https://github.com/ZakariaRochdi)*"
        )

        return top_k, threshold


# ─── Application principale ────────────────────────────────────────
def main():
    """Point d'entrée de l'application Streamlit."""
    # Charger la configuration
    config = load_app_config()
    setup_page(config)
    inject_css()

    # Header
    display_header(config)

    # Sidebar
    top_k, threshold = display_sidebar(config)

    # Initialiser les composants RAG (en cache)
    @st.cache_resource
    def init_retriever():
        return SemanticRetriever(config)

    @st.cache_resource
    def init_generator():
        return ResponseGenerator(config)

    # Initialisation
    try:
        retriever = init_retriever()
        generator = init_generator()
        system_ready = True
    except Exception as e:
        st.error(
            f"⚠️ Erreur lors de l'initialisation du système : {e}\n\n"
            "Vérifiez que l'index vectoriel a été construit "
            "(python main.py ingest && python main.py index)."
        )
        system_ready = False

    # Historique des conversations
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Afficher l'historique
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                display_response(msg["response"])

    # Zone de saisie
    if question := st.chat_input(
        "Posez votre question juridique...",
        disabled=not system_ready,
    ):
        # Afficher la question
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })
        with st.chat_message("user"):
            st.write(question)

        # Recherche + Génération
        with st.chat_message("assistant"):
            with st.spinner("🔍 Recherche dans le corpus juridique..."):
                results = retriever.search(question, top_k=top_k, threshold=threshold)

            with st.spinner("📝 Génération de la réponse..."):
                response = generator.generate(question, results)

            # Affichage
            display_response(response)
            display_sources_detail(results)

            # Sauvegarder dans l'historique
            st.session_state.messages.append({
                "role": "assistant",
                "response": response,
                "results": results,
            })

            # Limiter l'historique
            max_history = config.get("ui", {}).get("max_history", 10) * 2
            if len(st.session_state.messages) > max_history:
                st.session_state.messages = st.session_state.messages[-max_history:]


if __name__ == "__main__":
    main()
