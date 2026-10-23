"""App entry: settings → knowledge base → retriever + LLM → RAG → Gradio page (`python -m persona`)."""

import logging

import spaces

from persona import config, knowledge_base, ui
from persona.llm import LLM
from persona.rag import Rag
from persona.retriever import Retriever

log = logging.getLogger("persona")


@spaces.GPU(duration=1)
def zerogpu_startup_check() -> None:
    """Never called: ZeroGPU refuses to start without one @spaces.GPU function (Step 1). The app runs on CPU."""


def build_page():
    """The chat page, or — when the knowledge base or the LLM settings can't be loaded — the 'unavailable' page."""
    try:
        cfg = config.load()
        chunks = knowledge_base.load(cfg)
        rag = Rag(Retriever(chunks), LLM(cfg.llm_endpoints), chunks)
    except Exception:  # any startup failure: show a clear page instead of crashing the Space
        log.exception("startup failed; serving the 'unavailable' page")
        return ui.create_unavailable()
    models = [m for e in cfg.llm_endpoints for m in e.models]
    log.info("ready: %d chunks, models in order: %s", len(chunks), ", ".join(models))
    return ui.create(rag.answer, model_name=models[0])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # keep the Space log to the app's own lines
    # ssr_mode=False: with server-side rendering the page showed before it reacted, losing early clicks (Step 5)
    build_page().launch(ssr_mode=False)


if __name__ == "__main__":
    main()
