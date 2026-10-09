"""
tests/test_generation.py — Tests unitaires du module de génération
===================================================================
Teste les fonctionnalités du générateur :
  - Construction des prompts
  - Gestion de l'abstention
  - Format des réponses
  - Citations des sources
  - Mode fallback
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def mock_config():
    """Configuration de test."""
    return {
        "generation": {
            "provider": "ollama",
            "model_name": "mistral",
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 0.9,
            "abstention_threshold": 0.25,
            "system_prompt": "Tu es un assistant juridique.",
        },
    }


@pytest.fixture
def sample_results():
    """Résultats de recherche simulés."""
    return [
        {
            "text": "Article 152 : La salariée a droit à un congé de maternité de quatorze semaines.",
            "score": 0.92,
            "metadata": {
                "source": "code-travail-maroc.pdf",
                "article": "Article 152",
            },
            "chunk_id": 0,
        },
        {
            "text": "Article 153 : La salariée peut demander une prolongation de congé non payé.",
            "score": 0.78,
            "metadata": {
                "source": "code-travail-maroc.pdf",
                "article": "Article 153",
            },
            "chunk_id": 1,
        },
    ]


@pytest.fixture
def low_score_results():
    """Résultats avec des scores en dessous du seuil d'abstention."""
    return [
        {
            "text": "Texte peu pertinent.",
            "score": 0.15,
            "metadata": {
                "source": "code-travail-maroc.pdf",
                "article": "Article 1",
            },
            "chunk_id": 0,
        },
        {
            "text": "Autre texte peu pertinent.",
            "score": 0.10,
            "metadata": {
                "source": "code-travail-maroc.pdf",
                "article": "Article 2",
            },
            "chunk_id": 1,
        },
    ]


@pytest.fixture
def generator(mock_config):
    """Instance du générateur avec client mock."""
    with patch("generation.generateur.ResponseGenerator._init_llm"):
        from generation.generateur import ResponseGenerator
        gen = ResponseGenerator(mock_config)
        gen.client = None  # Mode fallback
        return gen


# ─── Tests de la génération ────────────────────────────────────────

class TestResponseGenerator:
    """Tests pour la classe ResponseGenerator."""

    def test_generate_returns_expected_keys(self, generator, sample_results):
        """Vérifie que la réponse contient toutes les clés attendues."""
        response = generator.generate(
            "Quelle est la durée du congé de maternité ?",
            sample_results,
        )

        assert "answer" in response
        assert "sources" in response
        assert "abstained" in response
        assert "scores" in response

    def test_generate_includes_sources(self, generator, sample_results):
        """Vérifie que les sources sont incluses dans la réponse."""
        response = generator.generate(
            "Quelle est la durée du congé de maternité ?",
            sample_results,
        )

        assert len(response["sources"]) > 0
        assert any("Article 152" in s for s in response["sources"])

    def test_generate_not_abstained_with_good_results(
        self, generator, sample_results
    ):
        """Vérifie que le système ne s'abstient pas avec de bons résultats."""
        response = generator.generate(
            "Quelle est la durée du congé de maternité ?",
            sample_results,
        )

        assert response["abstained"] is False

    def test_generate_abstains_with_empty_results(self, generator):
        """Vérifie l'abstention quand aucun résultat n'est trouvé."""
        response = generator.generate(
            "Comment fonctionne la retraite ?",
            [],  # Pas de résultats
        )

        assert response["abstained"] is True
        assert "ne dispose pas" in response["answer"].lower() or \
               "pas de cette information" in response["answer"].lower()
        assert response["sources"] == []

    def test_generate_abstains_with_low_scores(
        self, generator, low_score_results
    ):
        """Vérifie l'abstention quand les scores sont trop bas."""
        response = generator.generate(
            "Question hors sujet",
            low_score_results,
        )

        assert response["abstained"] is True

    def test_generate_scores_returned(self, generator, sample_results):
        """Vérifie que les scores des chunks sont retournés."""
        response = generator.generate(
            "congé maternité",
            sample_results,
        )

        assert len(response["scores"]) == len(sample_results)
        assert response["scores"][0] == 0.92


class TestPromptBuilding:
    """Tests pour la construction des prompts."""

    def test_build_prompt_contains_question(self, generator):
        """Vérifie que le prompt contient la question."""
        question = "Quelle est la durée du congé ?"
        prompt = generator._build_prompt(
            question,
            "Contexte test",
            ["Source 1"],
        )

        assert question in prompt

    def test_build_prompt_contains_context(self, generator):
        """Vérifie que le prompt contient le contexte."""
        context = "Article 152 : congé de maternité de 14 semaines."
        prompt = generator._build_prompt(
            "Question test",
            context,
            ["Code du travail, Article 152"],
        )

        assert context in prompt

    def test_build_prompt_contains_sources(self, generator):
        """Vérifie que le prompt contient les sources."""
        sources = ["Code du travail, Article 152", "Code du travail, Article 153"]
        prompt = generator._build_prompt(
            "Question test",
            "Contexte test",
            sources,
        )

        for source in sources:
            assert source in prompt


class TestAbstention:
    """Tests de la logique d'abstention."""

    def test_should_abstain_empty_results(self, generator):
        """Abstention avec liste vide."""
        assert generator._should_abstain([]) is True

    def test_should_abstain_low_scores(self, generator, low_score_results):
        """Abstention avec scores bas."""
        assert generator._should_abstain(low_score_results) is True

    def test_should_not_abstain_good_scores(self, generator, sample_results):
        """Pas d'abstention avec de bons scores."""
        assert generator._should_abstain(sample_results) is False

    def test_abstention_threshold_boundary(self, generator):
        """Test du seuil exact d'abstention."""
        # Score moyen = 0.25 = seuil → devrait s'abstenir (score < seuil est strict)
        boundary_results = [
            {"score": 0.25, "text": "test", "metadata": {}},
        ]
        # 0.25 n'est pas < 0.25, donc ne devrait PAS s'abstenir
        assert generator._should_abstain(boundary_results) is False

        # Score juste en dessous du seuil
        below_results = [
            {"score": 0.24, "text": "test", "metadata": {}},
        ]
        assert generator._should_abstain(below_results) is True


class TestFallbackMode:
    """Tests du mode fallback (sans LLM)."""

    def test_fallback_response_contains_extracts(
        self, generator, sample_results
    ):
        """Vérifie que le fallback inclut les extraits."""
        response = generator._generate_fallback(
            "Question test",
            sample_results,
            ["Code du travail, Article 152"],
        )

        assert "extrait" in response.lower() or "Extrait" in response

    def test_fallback_response_contains_warning(
        self, generator, sample_results
    ):
        """Vérifie l'avertissement dans le fallback."""
        response = generator._generate_fallback(
            "Question test",
            sample_results,
            ["Code du travail, Article 152"],
        )

        assert "informatif" in response.lower()


class TestGeneratorInit:
    """Tests de l'initialisation du générateur."""

    def test_default_system_prompt(self, mock_config):
        """Vérifie le prompt système par défaut."""
        config_no_prompt = mock_config.copy()
        config_no_prompt["generation"] = {
            "provider": "ollama",
            "model_name": "mistral",
        }

        with patch("generation.generateur.ResponseGenerator._init_llm"):
            from generation.generateur import ResponseGenerator
            gen = ResponseGenerator(config_no_prompt)

        assert "juridique" in gen.system_prompt.lower()
        assert "marocain" in gen.system_prompt.lower()

    def test_custom_system_prompt(self, mock_config):
        """Vérifie qu'un prompt système personnalisé est utilisé."""
        with patch("generation.generateur.ResponseGenerator._init_llm"):
            from generation.generateur import ResponseGenerator
            gen = ResponseGenerator(mock_config)

        assert gen.system_prompt == "Tu es un assistant juridique."
