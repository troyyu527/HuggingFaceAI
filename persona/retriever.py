"""Vector search over the knowledge-base chunks: embed them once at startup, then cosine top-k per query."""

# ── LangChain version (active) ─────────────────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from persona.models import Chunk, ScoredChunk

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
TOP_K = 8  # 6 in Step 2; 8 since Step 7 (a leadership chunk ranked 8th)
SCORE_THRESHOLD = 0.5


def default_embeddings() -> HuggingFaceEmbeddings:
    """bge-base on CPU with normalized vectors, so the dot product is the cosine similarity."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class Retriever:
    """Holds the embedded chunks; `embeddings` needs embed_documents() and embed_query()."""

    def __init__(self, chunks: list[Chunk], embeddings=None, query_prefix: str = QUERY_PREFIX):
        self._chunks = {c.id: c for c in chunks}
        self._prefix = query_prefix
        self._store = InMemoryVectorStore(embeddings or default_embeddings())
        documents = [Document(page_content=c.text, metadata={"title": c.title}) for c in chunks]
        self._store.add_documents(documents, ids=[c.id for c in chunks])

    def search(self, query: str, k: int = TOP_K, threshold: float = SCORE_THRESHOLD) -> list[ScoredChunk]:
        """The k most similar chunks, best first, without those scoring below the threshold."""
        hits = self._store.similarity_search_with_score(self._prefix + query, k=k)
        return [ScoredChunk(self._chunks[doc.id], score) for doc, score in hits if score >= threshold]


# ── Plain Python version (no LangChain) ──────────────────────────────────────────────────────────────────
## Same public names and signatures: comment the block above and uncomment this one to run it.
# import numpy as np
# from sentence_transformers import SentenceTransformer
#
# from persona.models import Chunk, ScoredChunk
#
# EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge query instruction
# TOP_K = 8  # 6 in Step 2; 8 since Step 7 (a leadership chunk ranked 8th)
# SCORE_THRESHOLD = 0.5
#
#
# class SentenceEmbeddings:
#     """sentence-transformers behind the same two methods LangChain's Embeddings has."""
#
#     def __init__(self, model_name: str = EMBED_MODEL):
#         self._model = SentenceTransformer(model_name, device="cpu")
#
#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         return self._model.encode(texts, normalize_embeddings=True).tolist()
#
#     def embed_query(self, text: str) -> list[float]:
#         return self._model.encode(text, normalize_embeddings=True).tolist()
#
#
# def default_embeddings() -> SentenceEmbeddings:
#     """bge-base on CPU with normalized vectors, so the dot product is the cosine similarity."""
#     return SentenceEmbeddings()
#
#
# def _normalize(vectors: np.ndarray) -> np.ndarray:
#     return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
#
#
# class Retriever:
#     """Holds the embedded chunks; `embeddings` needs embed_documents() and embed_query()."""
#
#     def __init__(self, chunks: list[Chunk], embeddings=None, query_prefix: str = QUERY_PREFIX):
#         self._chunks = list(chunks)
#         self._prefix = query_prefix
#         self._embeddings = embeddings or default_embeddings()
#         self._vectors = _normalize(np.array(self._embeddings.embed_documents([c.text for c in chunks])))
#
#     def search(self, query: str, k: int = TOP_K, threshold: float = SCORE_THRESHOLD) -> list[ScoredChunk]:
#         """The k most similar chunks, best first, without those scoring below the threshold."""
#         scores = self._vectors @ _normalize(np.array(self._embeddings.embed_query(self._prefix + query)))
#         best = np.argsort(scores)[::-1][:k]
#         return [ScoredChunk(self._chunks[i], float(scores[i])) for i in best if scores[i] >= threshold]
