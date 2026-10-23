"""Load the knowledge-base chunks: a local chunks.jsonl in development, the private HF Dataset in the Space."""

from pathlib import Path

from huggingface_hub import hf_hub_download

from persona.config import Config
from persona.models import Chunk, parse_chunk_line

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_FILE = "chunks.jsonl"


def chunks_file(cfg: Config) -> Path:
    """CHUNKS_PATH if set (relative paths start at the project root), else download from the private Dataset."""
    if cfg.chunks_path:
        path = Path(cfg.chunks_path)
        return path if path.is_absolute() else PROJECT_ROOT / path
    if not (cfg.kb_dataset and cfg.hf_token):
        raise RuntimeError("Set CHUNKS_PATH, or KB_DATASET and HF_TOKEN, to load the knowledge base")
    return Path(hf_hub_download(cfg.kb_dataset, DATASET_FILE, repo_type="dataset", token=cfg.hf_token))


def load(cfg: Config) -> list[Chunk]:
    """Read every chunk; raise ValueError on a bad line or a duplicate id."""
    path = chunks_file(cfg)
    chunks = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    chunks.append(parse_chunk_line(line))
                except ValueError as e:
                    raise ValueError(f"{path.name} line {number}: {e}") from e
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate chunk ids")
    return chunks
