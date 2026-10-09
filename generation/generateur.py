"""
generation/generateur.py — Génération de réponses juridiques sourcées
======================================================================
Étape 5 du pipeline : Génération de la réponse.

Le générateur :
  1. Construit le contexte à partir des chunks récupérés
  2. Formule un prompt structuré pour le LLM
  3. Génère une réponse claire et sourcée
  4. Gère l'abstention ("Je ne sais pas") quand le contexte est insuffisant
"""

from typing import List, Dict, Any, Optional

from loguru import logger


class ResponseGenerator:
    """
    Générateur de réponses juridiques basé sur un LLM local.

    Utilise Ollama (Mistral) ou HuggingFace Transformers
    pour générer des réponses ancrées dans le corpus juridique.
    """

    def __init__(self, config: dict):
        """
        Initialise le générateur avec la configuration.

        Args:
            config: Configuration globale du projet.
        """
        self.config = config
        self.gen_config = config.get("generation", {})

        self.provider = self.gen_config.get("provider", "ollama")
        self.model_name = self.gen_config.get("model_name", "mistral")
        self.temperature = self.gen_config.get("temperature", 0.1)
        self.max_tokens = self.gen_config.get("max_tokens", 1024)
        self.top_p = self.gen_config.get("top_p", 0.9)
        self.abstention_threshold = self.gen_config.get(
            "abstention_threshold", 0.25
        )

        self.system_prompt = self.gen_config.get("system_prompt", self._default_system_prompt())

        # Initialiser le client LLM
        self._init_llm()

    def _default_system_prompt(self) -> str:
        """Prompt système par défaut."""
        return (
            "Tu es un assistant juridique spécialisé dans le droit marocain. "
            "Tu réponds aux questions des citoyens en te basant UNIQUEMENT sur "
            "les extraits de textes juridiques qui te sont fournis comme contexte.\n\n"
            "Règles strictes :\n"
            "1. Réponds UNIQUEMENT à partir du contexte fourni.\n"
            "2. Cite TOUJOURS la source exacte (loi, article, code).\n"
            "3. Si le contexte ne contient pas l'information, dis clairement : "
            '"Je ne dispose pas de cette information dans les textes consultés."\n'
            "4. Formule ta réponse de manière claire et accessible au citoyen.\n"
            '5. Ajoute un avertissement : "Réponse fournie à titre informatif ; '
            'pour une situation particulière, consultez un professionnel."'
        )

    def _init_llm(self) -> None:
        """Initialise le client LLM selon le provider configuré."""
        if self.provider == "ollama":
            try:
                import ollama
                self.client = ollama
                logger.info(
                    f"🤖 LLM initialisé : Ollama ({self.model_name})"
                )
            except ImportError:
                logger.warning(
                    "Package ollama non installé. "
                    "Utilisation du mode fallback (template de réponse)."
                )
                self.client = None

        elif self.provider == "huggingface":
            try:
                from transformers import pipeline as hf_pipeline
                self.hf_pipe = hf_pipeline(
                    "text-generation",
                    model=self.model_name,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                self.client = self.hf_pipe
                logger.info(
                    f"🤖 LLM initialisé : HuggingFace ({self.model_name})"
                )
            except ImportError:
                logger.warning(
                    "Package transformers non installé. Mode fallback activé."
                )
                self.client = None
        else:
            logger.warning(f"Provider inconnu : {self.provider}. Mode fallback.")
            self.client = None

    def _build_prompt(
        self,
        question: str,
        context: str,
        sources: List[str],
    ) -> str:
        """
        Construit le prompt pour le LLM.

        Args:
            question: Question de l'utilisateur.
            context: Contexte agrégé des chunks pertinents.
            sources: Liste des sources identifiées.

        Returns:
            Prompt complet formaté.
        """
        prompt = (
            f"Contexte juridique :\n"
            f"---\n"
            f"{context}\n"
            f"---\n\n"
            f"Sources disponibles : {', '.join(sources)}\n\n"
            f"Question du citoyen : {question}\n\n"
            f"Réponse :"
        )
        return prompt

    def _should_abstain(self, results: List[Dict[str, Any]]) -> bool:
        """
        Détermine si le système doit s'abstenir de répondre.

        Critères d'abstention :
          - Aucun résultat pertinent
          - Score moyen en dessous du seuil
        """
        if not results:
            return True

        avg_score = sum(r["score"] for r in results) / len(results)

        if avg_score < self.abstention_threshold:
            logger.info(
                f"⚠️ Score moyen ({avg_score:.4f}) sous le seuil "
                f"({self.abstention_threshold}). Abstention recommandée."
            )
            return True

        return False

    def _generate_with_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Génère une réponse avec Ollama."""
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
        )
        return response["message"]["content"]

    def _generate_with_huggingface(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        """Génère une réponse avec HuggingFace Transformers."""
        full_prompt = f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        result = self.client(full_prompt)
        return result[0]["generated_text"].split("[/INST]")[-1].strip()

    def _generate_fallback(
        self,
        question: str,
        results: List[Dict[str, Any]],
        sources: List[str],
    ) -> str:
        """
        Génère une réponse de base sans LLM (mode fallback).

        Retourne directement les extraits pertinents avec les sources.
        """
        response_parts = [
            "📋 Voici les extraits les plus pertinents trouvés dans le corpus juridique :\n"
        ]

        for i, r in enumerate(results[:3]):
            article = r["metadata"].get("article", "")
            source = r["metadata"].get("source", "")
            response_parts.append(
                f"\n📎 Extrait {i + 1} ({source}, {article}) :\n"
                f"{r['text'][:500]}"
            )

        response_parts.append(f"\n\n📌 Sources : {', '.join(sources)}")
        response_parts.append(
            "\n⚠️ Réponse fournie à titre informatif ; "
            "pour une situation particulière, consultez un professionnel."
        )

        return "\n".join(response_parts)

    def generate(
        self,
        question: str,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Génère une réponse complète à partir de la question et des résultats.

        Args:
            question: Question de l'utilisateur en langage naturel.
            results: Résultats de la recherche sémantique (retriever).

        Returns:
            Dictionnaire contenant :
              - answer: texte de la réponse
              - sources: liste des sources citées
              - abstained: booléen indiquant si le système s'est abstenu
              - scores: scores de similarité des chunks utilisés
        """
        logger.info(f"📝 Génération de la réponse pour : '{question[:80]}...'")

        # Vérifier l'abstention
        if self._should_abstain(results):
            logger.info("🚫 Abstention : contexte insuffisant")
            return {
                "answer": (
                    "Je ne dispose pas de cette information dans les textes "
                    "juridiques consultés. Je vous recommande de consulter un "
                    "professionnel du droit pour obtenir une réponse fiable à "
                    "votre question."
                ),
                "sources": [],
                "abstained": True,
                "scores": [],
            }

        # Extraire les sources
        sources = []
        for r in results:
            source = r["metadata"].get("source", "")
            article = r["metadata"].get("article", "")
            ref = f"{source}, {article}" if article else source
            if ref and ref not in sources:
                sources.append(ref)

        # Construire le contexte
        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(f"[Extrait {i + 1}] :\n{r['text']}")
        context = "\n\n---\n\n".join(context_parts)

        # Générer la réponse
        user_prompt = self._build_prompt(question, context, sources)

        try:
            if self.client is not None:
                if self.provider == "ollama":
                    answer = self._generate_with_ollama(
                        self.system_prompt, user_prompt
                    )
                elif self.provider == "huggingface":
                    answer = self._generate_with_huggingface(
                        self.system_prompt, user_prompt
                    )
                else:
                    answer = self._generate_fallback(question, results, sources)
            else:
                answer = self._generate_fallback(question, results, sources)

        except Exception as e:
            logger.error(f"Erreur lors de la génération : {e}")
            logger.info("Basculement vers le mode fallback")
            answer = self._generate_fallback(question, results, sources)

        logger.success("✅ Réponse générée avec succès")

        return {
            "answer": answer,
            "sources": sources,
            "abstained": False,
            "scores": [r["score"] for r in results],
        }


if __name__ == "__main__":
    import yaml

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    generator = ResponseGenerator(config)

    # Test avec des résultats simulés
    fake_results = [
        {
            "text": "Article 152 : La salariée a droit à un congé de maternité de quatorze semaines.",
            "score": 0.92,
            "metadata": {
                "source": "code-travail-maroc.pdf",
                "article": "Article 152",
            },
        }
    ]

    response = generator.generate(
        "Quelle est la durée du congé de maternité ?",
        fake_results,
    )

    print(f"\n📋 Réponse : {response['answer']}")
    print(f"📌 Sources : {response['sources']}")
    print(f"🚫 Abstention : {response['abstained']}")
