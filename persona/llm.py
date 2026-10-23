"""Chat model: streamed and one-shot answers, trying the main provider's models first, then the fallback's."""

# ── LangChain version (active) ─────────────────────────────────────────────────────────────────────────
import logging
from collections.abc import Iterator

from langchain_openai import ChatOpenAI
from openai import OpenAIError

from persona.models import Endpoint, LLMUnavailable, plan_attempts

TEMPERATURE = 0.2
MAX_TOKENS = 600  # gpt-oss spends part of it on short reasoning; 400 cut answers off (Step 7)
TIMEOUT_SECONDS = 60

log = logging.getLogger("persona")


class LLM:
    """Messages are role/content dicts; `model` is the id of the model that answered last.

    A model that fails before any text arrives (quota used up, rate limit, outage) is skipped and the next one is
    tried; only when every model fails is LLMUnavailable raised. A failure after text has appeared can't be
    continued on another model, so it raises at once.
    """

    def __init__(self, endpoints: list[Endpoint]):
        attempts = plan_attempts(endpoints)
        if not attempts:
            raise ValueError("no LLM configured: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL")
        self.model: str | None = None
        self._attempts = [
            (
                model,
                ChatOpenAI(
                    base_url=endpoint.base_url,
                    api_key=endpoint.api_key,
                    model=model,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    timeout=TIMEOUT_SECONDS,
                    max_retries=1
                    if i == len(attempts) - 1
                    else 0,  # move on quickly; retry only the last one
                    extra_body=options,
                ),
            )
            for i, (endpoint, model, options) in enumerate(attempts)
        ]

    def _stream_one(self, attempt, messages: list[dict]) -> Iterator[tuple[str, str | None]]:
        """(text piece, model id reported by the provider) from one model."""
        _, chat = attempt
        for chunk in chat.stream(messages):
            yield chunk.content, chunk.response_metadata.get("model_name")

    def _complete_one(self, attempt, messages: list[dict], max_tokens: int) -> tuple[str, str | None]:
        _, chat = attempt
        reply = chat.invoke(messages, max_tokens=max_tokens)
        return reply.content, reply.response_metadata.get("model_name")

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield the answer piece by piece; raise LLMUnavailable if no model answers."""
        failures = []
        for attempt in self._attempts:
            self.model, got_text = attempt[0], False
            try:
                for piece, model_id in self._stream_one(attempt, messages):
                    self.model = model_id or self.model
                    if piece:
                        got_text = True
                        yield piece
            except OpenAIError as e:
                if got_text:
                    raise LLMUnavailable(f"{attempt[0]} failed mid-answer: {e}") from e
                failures.append(f"{attempt[0]}: {e}")
                log.warning("%s unavailable, trying the next model: %s", attempt[0], e)
                continue
            if got_text:
                return
            failures.append(f"{attempt[0]}: empty answer")
        raise LLMUnavailable("; ".join(failures))

    def complete(self, messages: list[dict], max_tokens: int = MAX_TOKENS) -> str:
        """The whole answer at once (used for the query rewrite)."""
        failures = []
        for attempt in self._attempts:
            try:
                text, model_id = self._complete_one(attempt, messages, max_tokens)
            except OpenAIError as e:
                failures.append(f"{attempt[0]}: {e}")
                log.warning("%s unavailable, trying the next model: %s", attempt[0], e)
                continue
            self.model = model_id or attempt[0]
            return text
        raise LLMUnavailable("; ".join(failures))


# ── Plain Python version (no LangChain) ──────────────────────────────────────────────────────────────────
## Same public names and signatures: comment the block above and uncomment this one to run it.
# import logging
# from collections.abc import Iterator
#
# from openai import OpenAI, OpenAIError
#
# from persona.models import Endpoint, LLMUnavailable, plan_attempts
#
# TEMPERATURE = 0.2
# MAX_TOKENS = 600  # gpt-oss spends part of it on short reasoning; 400 cut answers off (Step 7)
# TIMEOUT_SECONDS = 60
#
# log = logging.getLogger("persona")
#
#
# class LLM:
#     """Messages are role/content dicts; `model` is the id of the model that answered last.
#
#     A model that fails before any text arrives (quota used up, rate limit, outage) is skipped and the next one is
#     tried; only when every model fails is LLMUnavailable raised. A failure after text has appeared can't be
#     continued on another model, so it raises at once.
#     """
#
#     def __init__(self, endpoints: list[Endpoint]):
#         attempts = plan_attempts(endpoints)
#         if not attempts:
#             raise ValueError("no LLM configured: set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL")
#         self.model: str | None = None
#         clients = {}
#         self._attempts = []
#         for i, (endpoint, model, options) in enumerate(attempts):
#             retries = 1 if i == len(attempts) - 1 else 0  # move on quickly; retry only the last one
#             key = (endpoint.base_url, retries)
#             if key not in clients:
#                 clients[key] = OpenAI(
#                     base_url=endpoint.base_url,
#                     api_key=endpoint.api_key,
#                     timeout=TIMEOUT_SECONDS,
#                     max_retries=retries,
#                 )
#             self._attempts.append((model, clients[key], options))
#
#     def _create(self, attempt, messages: list[dict], max_tokens: int, stream: bool):
#         model, client, options = attempt
#         return client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=TEMPERATURE,
#             max_tokens=max_tokens,
#             stream=stream,
#             extra_body=options,
#         )
#
#     def _stream_one(self, attempt, messages: list[dict]) -> Iterator[tuple[str, str | None]]:
#         """(text piece, model id reported by the provider) from one model."""
#         for chunk in self._create(attempt, messages, MAX_TOKENS, stream=True):
#             piece = chunk.choices[0].delta.content if chunk.choices else None
#             yield piece or "", chunk.model
#
#     def _complete_one(self, attempt, messages: list[dict], max_tokens: int) -> tuple[str, str | None]:
#         reply = self._create(attempt, messages, max_tokens, stream=False)
#         return reply.choices[0].message.content or "", reply.model
#
#     def stream(self, messages: list[dict]) -> Iterator[str]:
#         """Yield the answer piece by piece; raise LLMUnavailable if no model answers."""
#         failures = []
#         for attempt in self._attempts:
#             self.model, got_text = attempt[0], False
#             try:
#                 for piece, model_id in self._stream_one(attempt, messages):
#                     self.model = model_id or self.model
#                     if piece:
#                         got_text = True
#                         yield piece
#             except OpenAIError as e:
#                 if got_text:
#                     raise LLMUnavailable(f"{attempt[0]} failed mid-answer: {e}") from e
#                 failures.append(f"{attempt[0]}: {e}")
#                 log.warning("%s unavailable, trying the next model: %s", attempt[0], e)
#                 continue
#             if got_text:
#                 return
#             failures.append(f"{attempt[0]}: empty answer")
#         raise LLMUnavailable("; ".join(failures))
#
#     def complete(self, messages: list[dict], max_tokens: int = MAX_TOKENS) -> str:
#         """The whole answer at once (used for the query rewrite)."""
#         failures = []
#         for attempt in self._attempts:
#             try:
#                 text, model_id = self._complete_one(attempt, messages, max_tokens)
#             except OpenAIError as e:
#                 failures.append(f"{attempt[0]}: {e}")
#                 log.warning("%s unavailable, trying the next model: %s", attempt[0], e)
#                 continue
#             self.model = model_id or attempt[0]
#             return text
#         raise LLMUnavailable("; ".join(failures))
