"""Entry 2: upload the local chunks.jsonl to the private HF Dataset (`uv run python -m persona.upload_kb`)."""

import sys

from huggingface_hub import HfApi

from persona import config, knowledge_base


def main() -> int:
    cfg = config.load()
    if not (cfg.chunks_path and cfg.kb_dataset and cfg.hf_token):
        print("Set CHUNKS_PATH, KB_DATASET and HF_TOKEN (a write token) in .env first.")
        return 1
    chunks = knowledge_base.load(cfg)  # validates every line before anything is uploaded
    api = HfApi(token=cfg.hf_token)
    if not api.repo_info(cfg.kb_dataset, repo_type="dataset").private:
        print(f"Refusing to upload: {cfg.kb_dataset} is not private.")
        return 1
    commit = api.upload_file(
        path_or_fileobj=str(knowledge_base.chunks_file(cfg)),
        path_in_repo=knowledge_base.DATASET_FILE,
        repo_id=cfg.kb_dataset,
        repo_type="dataset",
        commit_message=f"Update knowledge base ({len(chunks)} chunks)",
    )
    print(f"Uploaded {len(chunks)} chunks to {cfg.kb_dataset} (private): {commit.commit_url}")
    print("Restart the Space (Settings → Restart) to load them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
