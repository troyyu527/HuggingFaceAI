"""answer(question, history): rewrite → search original + rewritten → prompt with context → streamed answer."""

# ── LangChain version (active) ─────────────────────────────────────────────────────────────────────────
import logging
import time
from collections.abc import Iterator

from langchain_core.messages import convert_to_openai_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from persona.models import (
    BUSY_MESSAGE,
    REWRITE_PROMPT,
    SUMMARY_CHUNK_ID,
    SYSTEM_PROMPT,
    AnswerUpdate,
    Chunk,
    LLMUnavailable,
    ScoredChunk,
    clean_rewrite,
    extract_sources_media,
    format_context,
    format_history,
    merge_chunks,
    search_query,
    trim_history,
    validate_question,
)
from persona.retriever import TOP_K

log = logging.getLogger("persona")
REWRITE_MAX_TOKENS = 60

_ANSWER_TEMPLATE = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), MessagesPlaceholder("history"), ("human", "{question}")]
)
_REWRITE_TEMPLATE = ChatPromptTemplate.from_messages([("human", REWRITE_PROMPT)])


def build_messages(
    question: str, history: list[dict], context: list[ScoredChunk], summary: str
) -> list[dict]:
    """System prompt (role · summary · rules · context) + clean history + the question."""
    prompt = _ANSWER_TEMPLATE.invoke(
        {"summary": summary, "context": format_context(context), "history": history, "question": question}
    )
    return convert_to_openai_messages(prompt.to_messages())


def build_rewrite_messages(question: str, history: list[dict]) -> list[dict]:
    prompt = _REWRITE_TEMPLATE.invoke({"history": format_history(history), "question": question})
    return convert_to_openai_messages(prompt.to_messages())


class Rag:
    """Ties the retriever and the LLM together; `chunks` supplies the always-on summary (profile#00)."""

    def __init__(self, retriever, llm, chunks: list[Chunk]):
        self._retriever = retriever
        self._llm = llm
        self.summary = next((c.text for c in chunks if c.id == SUMMARY_CHUNK_ID), "")

    def rewrite_query(self, question: str, history: list[dict]) -> str:
        """One short LLM call; the original question if it fails."""
        try:
            reply = self._llm.complete(
                build_rewrite_messages(question, history), max_tokens=REWRITE_MAX_TOKENS
            )
        except LLMUnavailable as e:
            log.warning("rewrite failed, using the original question: %s", e)
            return question
        return clean_rewrite(reply, question)

    def answer(self, question: str, history: list[dict]) -> Iterator[AnswerUpdate]:
        """Stream AnswerUpdates; the last one has done=True, the model id and the seconds taken."""
        start = time.time()
        problem = validate_question(question)
        if problem:
            yield AnswerUpdate(text=problem, done=True)
            return
        history = trim_history(history)
        # rewrite only follow-ups: a first question has nothing to resolve (Q3.1, saves one LLM call)
        query = self.rewrite_query(question, history) if history else question
        found = self._retriever.search(search_query(question))
        if query != question:
            found = merge_chunks(found, self._retriever.search(search_query(query)), k=TOP_K)
        context = [s for s in found if s.chunk.id != SUMMARY_CHUNK_ID]  # the summary is already in the prompt
        media = extract_sources_media([s.chunk for s in found])
        messages = build_messages(question, history, context, self.summary)
        text = ""
        try:
            for piece in self._llm.stream(messages):
                text += piece
                yield AnswerUpdate(text=text, sources=found, media=media, query=query)
        except LLMUnavailable as e:
            log.warning("answer failed after %.1fs: %s", time.time() - start, e)
            yield AnswerUpdate(
                BUSY_MESSAGE, found, media, query, None, time.time() - start, done=True, failed=True
            )
            return
        seconds = time.time() - start
        log.info("answered by %s in %.1fs (%d sources)", self._llm.model, seconds, len(found))
        yield AnswerUpdate(text, found, media, query, self._llm.model, seconds, done=True)


# ── Plain Python version (no LangChain) ──────────────────────────────────────────────────────────────────
## Same public names and signatures: comment the block above and uncomment this one to run it.
# import logging
# import time
# from collections.abc import Iterator
#
# from persona.models import (
#     BUSY_MESSAGE,
#     REWRITE_PROMPT,
#     SUMMARY_CHUNK_ID,
#     SYSTEM_PROMPT,
#     AnswerUpdate,
#     Chunk,
#     LLMUnavailable,
#     ScoredChunk,
#     clean_rewrite,
#     extract_sources_media,
#     format_context,
#     format_history,
#     merge_chunks,
#     search_query,
#     trim_history,
#     validate_question,
# )
# from persona.retriever import TOP_K
#
# log = logging.getLogger("persona")
# REWRITE_MAX_TOKENS = 60
#
#
# def build_messages(
#     question: str, history: list[dict], context: list[ScoredChunk], summary: str
# ) -> list[dict]:
#     """System prompt (role · summary · rules · context) + clean history + the question."""
#     system = SYSTEM_PROMPT.format(summary=summary, context=format_context(context))
#     return [{"role": "system", "content": system}] + history + [{"role": "user", "content": question}]
#
#
# def build_rewrite_messages(question: str, history: list[dict]) -> list[dict]:
#     return [
#         {"role": "user", "content": REWRITE_PROMPT.format(history=format_history(history), question=question)}
#     ]
#
#
# class Rag:
#     """Ties the retriever and the LLM together; `chunks` supplies the always-on summary (profile#00)."""
#
#     def __init__(self, retriever, llm, chunks: list[Chunk]):
#         self._retriever = retriever
#         self._llm = llm
#         self.summary = next((c.text for c in chunks if c.id == SUMMARY_CHUNK_ID), "")
#
#     def rewrite_query(self, question: str, history: list[dict]) -> str:
#         """One short LLM call; the original question if it fails."""
#         try:
#             reply = self._llm.complete(
#                 build_rewrite_messages(question, history), max_tokens=REWRITE_MAX_TOKENS
#             )
#         except LLMUnavailable as e:
#             log.warning("rewrite failed, using the original question: %s", e)
#             return question
#         return clean_rewrite(reply, question)
#
#     def answer(self, question: str, history: list[dict]) -> Iterator[AnswerUpdate]:
#         """Stream AnswerUpdates; the last one has done=True, the model id and the seconds taken."""
#         start = time.time()
#         problem = validate_question(question)
#         if problem:
#             yield AnswerUpdate(text=problem, done=True)
#             return
#         history = trim_history(history)
#         # rewrite only follow-ups: a first question has nothing to resolve (Q3.1, saves one LLM call)
#         query = self.rewrite_query(question, history) if history else question
#         found = self._retriever.search(search_query(question))
#         if query != question:
#             found = merge_chunks(found, self._retriever.search(search_query(query)), k=TOP_K)
#         context = [s for s in found if s.chunk.id != SUMMARY_CHUNK_ID]  # the summary is already in the prompt
#         media = extract_sources_media([s.chunk for s in found])
#         messages = build_messages(question, history, context, self.summary)
#         text = ""
#         try:
#             for piece in self._llm.stream(messages):
#                 text += piece
#                 yield AnswerUpdate(text=text, sources=found, media=media, query=query)
#         except LLMUnavailable as e:
#             log.warning("answer failed after %.1fs: %s", time.time() - start, e)
#             yield AnswerUpdate(
#                 BUSY_MESSAGE, found, media, query, None, time.time() - start, done=True, failed=True
#             )
#             return
#         seconds = time.time() - start
#         log.info("answered by %s in %.1fs (%d sources)", self._llm.model, seconds, len(found))
#         yield AnswerUpdate(text, found, media, query, self._llm.model, seconds, done=True)
