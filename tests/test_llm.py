"""Tests for the LLM service (Groq API wrapper)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.exceptions import ExternalServiceError
from app.services.llm import (
    EvaluationResult,
    LLMService,
    QuestionResult,
    SummaryResult,
)


def _mock_completion(content: str) -> MagicMock:
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def llm_service() -> LLMService:
    """Create an LLMService with a mocked OpenAI client."""
    with patch("app.services.llm.OpenAI"):
        service = LLMService(api_key="test-key")
    return service


class TestGenerateSummary:
    """Tests for LLMService.generate_summary."""

    def test_valid_json(self, llm_service: LLMService) -> None:
        """Valid summary JSON should be parsed correctly."""
        payload = json.dumps(
            {"summary": "A concise summary.", "page_citations": [1, 3, 5]}
        )
        llm_service.client.chat.completions.create.return_value = _mock_completion(
            payload
        )

        result = llm_service.generate_summary("some context")

        assert isinstance(result, SummaryResult)
        assert result.summary == "A concise summary."
        assert result.page_citations == [1, 3, 5]

    def test_json_wrapped_in_code_fence(self, llm_service: LLMService) -> None:
        """JSON wrapped in markdown code fences should still parse."""
        payload = '```json\n{"summary": "Fenced.", "page_citations": [2]}\n```'
        llm_service.client.chat.completions.create.return_value = _mock_completion(
            payload
        )

        result = llm_service.generate_summary("context")
        assert result.summary == "Fenced."

    def test_json_with_extra_text_is_recovered(self, llm_service: LLMService) -> None:
        """Parser should extract JSON even if model adds surrounding text."""
        payload = (
            'Here is the result:\n{"summary":"Recovered","page_citations":[1,2]}\n'
            "Thanks!"
        )
        llm_service.client.chat.completions.create.return_value = _mock_completion(
            payload
        )

        result = llm_service.generate_summary("context")
        assert result.summary == "Recovered"
        assert result.page_citations == [1, 2]

    def test_trailing_commas_are_repaired(self, llm_service: LLMService) -> None:
        """Common JSON trailing-comma mistakes should be repaired."""
        payload = '{"summary":"Fixed commas","page_citations":[1,2,],}'
        llm_service.client.chat.completions.create.return_value = _mock_completion(
            payload
        )

        result = llm_service.generate_summary("context")
        assert result.summary == "Fixed commas"
        assert result.page_citations == [1, 2]

    def test_prompt_includes_context(self, llm_service: LLMService) -> None:
        """The user prompt sent to the LLM should contain the context."""
        valid = json.dumps({"summary": "ok", "page_citations": []})
        llm_service.client.chat.completions.create.return_value = _mock_completion(
            valid
        )

        llm_service.generate_summary("MY_SPECIAL_CONTEXT_TEXT")

        call_args = llm_service.client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "MY_SPECIAL_CONTEXT_TEXT" in user_msg


class TestGenerateQuestions:
    """Tests for LLMService.generate_questions."""

    def test_valid_json(self, llm_service: LLMService) -> None:
        """Valid questions JSON should return a list of QuestionResult."""
        payload = json.dumps(
            {
                "questions": [
                    {"question": "What is X?", "expected_answer": "X is Y."},
                    {"question": "Why Z?", "expected_answer": "Because W."},
                ]
            }
        )
        llm_service.client.chat.completions.create.return_value = _mock_completion(
            payload
        )

        result = llm_service.generate_questions("context", num_questions=2)

        assert len(result) == 2
        assert all(isinstance(q, QuestionResult) for q in result)
        assert result[0].question == "What is X?"

    def test_num_questions_in_prompt(self, llm_service: LLMService) -> None:
        """num_questions should appear in the user prompt."""
        valid = json.dumps({"questions": [{"question": "Q?", "expected_answer": "A."}]})
        llm_service.client.chat.completions.create.return_value = _mock_completion(
            valid
        )

        llm_service.generate_questions("context", num_questions=7)

        call_args = llm_service.client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "7" in user_msg


class TestEvaluateAnswer:
    """Tests for LLMService.evaluate_answer."""

    def test_valid_json(self, llm_service: LLMService) -> None:
        """Valid evaluation JSON should be parsed correctly."""
        payload = json.dumps({"score": 0.85, "feedback": "Good answer."})
        llm_service.client.chat.completions.create.return_value = _mock_completion(
            payload
        )

        result = llm_service.evaluate_answer(
            question="What?", expected_answer="This.", user_answer="That."
        )

        assert isinstance(result, EvaluationResult)
        assert result.score == 0.85
        assert result.feedback == "Good answer."


class TestRetryAndErrors:
    """Tests for retry logic and error propagation."""

    def test_retry_on_invalid_json(self, llm_service: LLMService) -> None:
        """First invalid response should trigger one retry, then succeed."""
        valid = json.dumps({"summary": "ok", "page_citations": []})
        llm_service.client.chat.completions.create.side_effect = [
            _mock_completion("NOT VALID JSON {{"),
            _mock_completion(valid),
        ]

        result = llm_service.generate_summary("context")

        assert result.summary == "ok"
        assert llm_service.client.chat.completions.create.call_count == 2

    def test_fails_after_retry_exhausted(self, llm_service: LLMService) -> None:
        """Two consecutive invalid responses should raise ExternalServiceError."""
        llm_service.client.chat.completions.create.side_effect = [
            _mock_completion("bad1"),
            _mock_completion("bad2"),
        ]

        with pytest.raises(ExternalServiceError):
            llm_service.generate_summary("context")

    def test_api_error_raises_external_service_error(
        self, llm_service: LLMService
    ) -> None:
        """API exception should be wrapped in ExternalServiceError."""
        from openai import OpenAIError

        llm_service.client.chat.completions.create.side_effect = OpenAIError(
            "connection refused"
        )

        with pytest.raises(ExternalServiceError) as exc_info:
            llm_service.generate_summary("context")

        assert "Groq" in exc_info.value.message

    def test_score_out_of_bounds_triggers_retry(self, llm_service: LLMService) -> None:
        """Score outside 0-1 should fail validation and trigger retry."""
        bad = json.dumps({"score": 1.5, "feedback": "Too high."})
        good = json.dumps({"score": 0.9, "feedback": "Fixed."})
        llm_service.client.chat.completions.create.side_effect = [
            _mock_completion(bad),
            _mock_completion(good),
        ]

        result = llm_service.evaluate_answer(
            question="Q?", expected_answer="A.", user_answer="B."
        )

        assert result.score == 0.9
        assert llm_service.client.chat.completions.create.call_count == 2

    def test_generate_questions_retry_on_invalid_schema(
        self, llm_service: LLMService
    ) -> None:
        """Invalid question schema should trigger one retry and recover."""
        bad = json.dumps({"questions": [{"question": "Missing expected answer"}]})
        good = json.dumps(
            {
                "questions": [
                    {
                        "question": "What is RAG?",
                        "expected_answer": "Retrieval-augmented generation.",
                    }
                ]
            }
        )
        llm_service.client.chat.completions.create.side_effect = [
            _mock_completion(bad),
            _mock_completion(good),
        ]

        result = llm_service.generate_questions("context", num_questions=1)

        assert len(result) == 1
        assert result[0].question == "What is RAG?"
        assert llm_service.client.chat.completions.create.call_count == 2
