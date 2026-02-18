"""LLM service wrapping Groq API (Llama 3.1 8B) via the OpenAI SDK."""

import json
import logging
import re

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field, ValidationError

from app.config import settings
from app.exceptions import ExternalServiceError

logger = logging.getLogger(__name__)


# region internal Pydantic response schemas


class SummaryResult(BaseModel):
    """Validated structure for a document summary."""

    summary: str
    page_citations: list[int]


class QuestionResult(BaseModel):
    """A single generated question with its expected answer."""

    question: str
    expected_answer: str


class QuestionsResult(BaseModel):
    """Wrapper for a list of generated questions."""

    questions: list[QuestionResult]


class EvaluationResult(BaseModel):
    """Validated structure for an answer evaluation."""

    score: float = Field(ge=0.0, le=1.0)
    feedback: str


# endregion

# region Prompts

_SUMMARY_SYSTEM = (
    "You are a document summarization assistant. "
    "Given document excerpts, produce a concise summary and list the page "
    "numbers referenced. Respond ONLY with valid JSON matching this schema:\n"
    '{"summary": "<string>", "page_citations": [<int>, ...]}'
)

_QUESTIONS_SYSTEM = (
    "You are a question generation assistant. "
    "Given document excerpts, generate study questions with expected answers. "
    "Respond ONLY with valid JSON matching this schema:\n"
    '{"questions": [{"question": "<string>", "expected_answer": "<string>"}, ...]}'
)

_EVALUATION_SYSTEM = (
    "You are an answer evaluation assistant. "
    "Compare the user's answer to the expected answer and score it. "
    "Respond ONLY with valid JSON matching this schema:\n"
    '{"score": <float 0.0-1.0>, "feedback": "<string>"}'
)

# endregion


class LLMService:
    """Centralized wrapper for all LLM interactions via the Groq API."""

    def __init__(
        self,
        api_key: str = settings.groq_api_key,
        model: str = settings.llm_model,
        temperature: float = settings.llm_temperature,
        max_tokens: int = settings.llm_max_tokens,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def generate_summary(self, context: str) -> SummaryResult:
        """Generate a structured summary from document context.

        Args:
            context: Concatenated chunk text to summarize.

        Returns:
            SummaryResult with summary text and page citations.
        """
        user_prompt = f"Summarize the following document excerpts:\n\n{context}"
        return self._call_and_parse(
            system_prompt=_SUMMARY_SYSTEM,
            user_prompt=user_prompt,
            model_class=SummaryResult,
        )

    def generate_questions(
        self, context: str, num_questions: int = 5
    ) -> list[QuestionResult]:
        """Generate study questions from document context.

        Args:
            context: Concatenated chunk text to base questions on.
            num_questions: Number of questions to generate.

        Returns:
            List of QuestionResult objects.
        """
        user_prompt = (
            f"Generate exactly {num_questions} study questions based on "
            f"the following document excerpts:\n\n{context}"
        )
        result = self._call_and_parse(
            system_prompt=_QUESTIONS_SYSTEM,
            user_prompt=user_prompt,
            model_class=QuestionsResult,
        )
        return result.questions

    def evaluate_answer(
        self,
        question: str,
        expected_answer: str,
        user_answer: str,
    ) -> EvaluationResult:
        """Evaluate a user's answer against the expected answer.

        Args:
            question: The original question.
            expected_answer: The reference answer.
            user_answer: The user-submitted answer to evaluate.

        Returns:
            EvaluationResult with score and feedback.
        """
        user_prompt = (
            f"Question: {question}\n"
            f"Expected answer: {expected_answer}\n"
            f"User's answer: {user_answer}\n\n"
            "Evaluate the user's answer."
        )
        return self._call_and_parse(
            system_prompt=_EVALUATION_SYSTEM,
            user_prompt=user_prompt,
            model_class=EvaluationResult,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make a single chat completion request to the Groq API.

        Raises:
            ExternalServiceError: If the API call fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except OpenAIError as exc:
            logger.error("Groq API call failed: %s", exc)
            raise ExternalServiceError(
                service="Groq",
                message=str(exc),
            ) from exc

    def _call_and_parse(
        self,
        system_prompt: str,
        user_prompt: str,
        model_class: type[BaseModel],
    ) -> BaseModel:
        """Call the LLM and parse the response, retrying once on failure.

        The method attempts to parse the raw response as JSON and validate it
        against the given Pydantic model. If parsing or validation fails, a
        single retry is made. If the retry also fails, an
        ExternalServiceError is raised.
        """
        raw = self._call_llm(system_prompt, user_prompt)

        try:
            return self._parse_json(raw, model_class)
        except (json.JSONDecodeError, ValidationError) as first_err:
            logger.warning("LLM returned invalid output, retrying once: %s", first_err)

        # Retry once
        raw = self._call_llm(system_prompt, user_prompt)
        try:
            return self._parse_json(raw, model_class)
        except (json.JSONDecodeError, ValidationError) as second_err:
            logger.error("LLM retry also failed: %s", second_err)
            raise ExternalServiceError(
                service="Groq",
                message=f"Invalid LLM output after retry: {second_err}",
            ) from second_err

    @staticmethod
    def _parse_json(raw: str, model_class: type[BaseModel]) -> BaseModel:
        """Extract and validate JSON from raw LLM text.

        Handles cases where the LLM wraps JSON in markdown code fences.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        parsed = json.loads(text)
        return model_class.model_validate(parsed)
