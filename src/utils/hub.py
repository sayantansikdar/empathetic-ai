"""Publish trained adapters (and their measured metrics) to the Hugging Face Hub."""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def write_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)


def push_adapter(local_dir: str, repo_id: Optional[str], subfolder: str, message: str) -> Optional[str]:
    """
    Upload an adapter folder to `repo_id/subfolder`. Uses the token from
    `huggingface-cli login` / `notebook_login()`; does nothing if repo_id is unset.
    Uploading right after each stage means a later Colab disconnect can't lose it.
    """
    if not repo_id:
        return None
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        path_in_repo=subfolder,
        commit_message=message,
        ignore_patterns=["checkpoint-*", "*.pt", "runs/*", "logs/*"],
    )
    url = f"https://huggingface.co/{repo_id}/tree/main/{subfolder}"
    logger.info(f"Pushed {local_dir} → {url}")
    print(f"  ⬆️  Uploaded to {url}")
    return url
